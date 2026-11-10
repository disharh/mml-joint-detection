"""
sim_lenses_4im_mml.py
======================

Simulate detection of quadruply-imaged (quad) strongly-lensed BBH mergers
that are *jointly* detectable by:

    * LVK (H1+L1+V1; all three detectors online; 4 lensed images required), and
    * Euclid VIS  (electromagnetic host-galaxy light).

This is the `mml` package equivalent of the old flat-script
``scripts/sim_lenses_4gw.py``. Where the original script imported loose
functions from ``scripts/*.py`` (``lens_mass.py``, ``lens_light.py``,
``bbh_pos.py``, ``gw_pop.py``, ``likelihood.py``, ...) and relied on
module-level globals, this version is built entirely on the
``src/mml`` package:

    mml.populations.Lens                 -- lens-galaxy sampler
    mml.populations.Source               -- source-galaxy sampler (MAF/denmarf)
    mml.populations.GWPopulation         -- BBH sampler + GW lensing
    mml.populations.BBHPositionSampler   -- BBH/source position sampler
    mml.lensing.einstein_radius          -- SIS Einstein radius
    mml.detectors.em.euclid.calculate_snr -- Euclid EM detectability
    mml.detectors.gw.lvk.LVKDetector     -- LVK network SNR + detection

Usage
-----
    python sim_lenses_4im_mml.py <job_id> [--n-systems N] [--out-dir DIR]

Each detected system is written as a YAML file under
``<out-dir>/quad_im/job_<job_id>/System_<n>_job<job_id>.yaml``, mirroring
the directory convention of the original ``sim_lenses_4gw.py``
(``lens_catalog/two_im/...``) but under a ``quad_im`` subdirectory since
this script requires 4 detected lensed images rather than 2.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

from mml.populations import (
    Lens,
    Source,
    GWPopulation,
    BBHPositionSampler,
)
from mml.lensing import einstein_radius
from mml.detectors.em.euclid import calculate_snr as calculate_euclid_snr, EUCLID
from mml.detectors.gw.lvk import LVKDetector
from mml.io.yaml import convert_dict



def log_cross_section(area: float) -> float:
    """
    Log probability that a random GW falls inside a caustic of a given
    area in the source plane, assuming a uniform distribution of GW
    sources over the sky.

    Equivalent to ``scripts/likelihood.py::lik_cross_sec``.

    Parameters
    ----------
    area : float
        Caustic area in the source plane [arcsec^2].

    Returns
    -------
    float
        log(area_in_steradians / 4*pi).
    """
    from astropy import units as u

    return float(
        np.log(
            area * (1 * u.arcsec).to_value(u.radian) ** 2 / (4 * np.pi)
        )
    )


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run(
    job_id: int,
    num_gw: int = 4,
    n_systems: int = 100,
    out_dir: str = "lens_catalog/quad_im",
    max_system_attempts: int = 100_000,
    max_lens_attempts: int = 1_000,
    max_gw_attempts: int = 2_000,
    theta_ein_min: float | None = None,
    seed_offset: int = 5678,
):
    """
    Simulate ``n_systems`` jointly EM+GW detected quad-image lens systems.

    Parameters
    ----------
    job_id : int
        Job/task index. Used to seed the RNG and to namespace output
        files, matching the original per-job cluster-array convention.

    num_gw : int
        Required number of detected lensed images (4 for a quad system).

    n_systems : int
        Number of detected systems to generate before stopping.

    out_dir : str
        Base output directory. Files are written under
        ``<out_dir>/job_<job_id>/``.

    max_system_attempts, max_lens_attempts, max_gw_attempts : int
        Retry budgets for the outer system loop, the lens-sampling loop,
        and the GW-sampling/detection loop respectively.

    theta_ein_min : float, optional
        Minimum Einstein radius [arcsec] required for a lens to be
        considered resolvable. Defaults to ``EUCLID.theta_ein_threshold``
        so the EM and lens-sampling cuts stay consistent.

    seed_offset : int
        Added to ``job_id`` to build the RNG seed, matching the original
        script's ``np.random.seed(5678 + job_id)`` convention.
    """

    if theta_ein_min is None:
        theta_ein_min = EUCLID.theta_ein_threshold

    rng = np.random.default_rng(seed_offset + job_id)

    # High-level samplers / detectors, all sharing one RNG where possible.
    lens_sampler = Lens(sigmazfn="cond_on_zs", rng=rng)
    source_sampler = Source(rng=rng)
    position_sampler = BBHPositionSampler(rng=rng)
    gw_population = GWPopulation(rng=rng)

    # Three-detector LVK network (H1, L1, V1), with every detector
    # online for every image. With duty cycles fixed to 1, the effective
    # network SNR is the quadrature sum of all three detector SNRs.
    lvk_detector = LVKDetector(
        num_detected_images=num_gw,
        min_detectors=3,
        duty_cycles={
            "H1": 1.0,
            "L1": 1.0,
            "V1": 1.0,
        },
    )

    n_det = 0
    system_attempts = 0
    tstart = time.perf_counter()

    job_out_dir = os.path.join(out_dir, f"job_{job_id}")

    while n_det < n_systems and system_attempts < max_system_attempts:
        system_attempts += 1
        system_start = time.perf_counter()
        timings = {}

        # ------------------------------------------------------------
        # SOURCE GALAXY SAMPLING
        # ------------------------------------------------------------
        t0 = time.perf_counter()
        source = source_sampler.sample(size=1)
        timings["source_sampling_time"] = time.perf_counter() - t0

        # ------------------------------------------------------------
        # LENS GALAXY SAMPLING (retry until resolvable)
        # ------------------------------------------------------------
        t0 = time.perf_counter()
        lens_attempts = 0
        lens_success = False
        lens = None

        while lens_attempts < max_lens_attempts:
            lens_attempts += 1

            lens = lens_sampler.sample(size=1, zs=source.z_source)

            theta_ein = einstein_radius(
                lens.sigma_lens,
                lens.z_lens,
                source.z_source,
            )

            if theta_ein >= theta_ein_min:
                lens.theta_ein = theta_ein
                lens_success = True
                break

        timings["lens_sampling_time"] = time.perf_counter() - t0
        timings["lens_attempts"] = lens_attempts

        if not lens_success:
            continue

        kwargs_lens = lens.to_lenstronomy()
        kwargs_source = source.to_lenstronomy()

        # ------------------------------------------------------------
        # BBH + SOURCE-GALAXY POSITION SAMPLING (within the quad caustic)
        # ------------------------------------------------------------
        try:
            position = position_sampler.sample(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                num_detected_gws=num_gw,
            )
        except Exception as e:
            print(f"Error while sampling BBH/source positions: {e}")
            continue

        # ------------------------------------------------------------
        # EM (EUCLID) DETECTION
        # ------------------------------------------------------------
        em_start = time.perf_counter()

        euclid_result = calculate_euclid_snr(
            lens,
            source,
            position,
        )

        em_detected = euclid_result.observable

        timings["em_detection_time"] = time.perf_counter() - em_start
        timings["em_attempts"] = 1

        if not em_detected:
            continue

        # ------------------------------------------------------------
        # GW SAMPLING + LVK DETECTION (retry until num_gw images detected)
        # ------------------------------------------------------------
        gw_start = time.perf_counter()
        gw_attempts = 0
        gw_success = False
        gw = None
        lensed = None
        lvk_result = None

        while gw_attempts < max_gw_attempts:
            gw_attempts += 1

            gw = gw_population.sample(size=1)

            try:
                lensed = gw_population.lens(
                    gw=gw,
                    lens=lens,
                    x_gw=position.x_gw,
                    y_gw=position.y_gw,
                    z_source=source.z_source,
                )
            except RuntimeError as e:
                # Lens equation solver produced an unusable image count
                # (0 images, or >5 images) for this random GW draw.
                continue

            if lensed.n_images < num_gw:
                continue

            lvk_result = lvk_detector.calculate_snr(
                gw=gw,
                lensed=lensed,
                z_source=source.z_source,
            )

            if lvk_result.detected:
                gw_success = True
                break

        timings["gw_sampling_detection_time"] = time.perf_counter() - gw_start
        timings["gw_attempts"] = gw_attempts

        if not gw_success:
            print(
                f"[DEBUG] EM detected but GW detection failed "
                f"at system_attempt={system_attempts}"
            )
            continue

        # ------------------------------------------------------------
        # SAVE DETECTED SYSTEM
        # ------------------------------------------------------------
        timings["total_system_time"] = time.perf_counter() - system_start
        timings["system_attempt"] = system_attempts

        n_det += 1
        print(f"System {n_det} detected!")

        gw_prms = gw.to_dict()
        gw_prms["x_gw"] = position.x_gw
        gw_prms["y_gw"] = position.y_gw
        gw_prms["caustic_area"] = position.caustic_area
        gw_prms["log_p_cross_sec"] = log_cross_section(position.caustic_area)

        # LVK detection summary. snr_net_ideal is the network SNR
        # before applying detector availability; snr_net is the SNR
        # after availability is applied. Here all detectors are always
        # online, so these two arrays should be identical.
        gw_prms["n_images_detected"] = lvk_result.n_detected
        gw_prms["event_detected"] = lvk_result.detected
        gw_prms["detected_indices"] = lvk_result.detected_indices
        gw_prms["detected_mask"] = lvk_result.detected_mask
        gw_prms["snr_net_ideal"] = lvk_result.snr_net_ideal
        gw_prms["snr_net"] = lvk_result.snr_net
        gw_prms["snr_H1"] = lvk_result.snr_H1
        gw_prms["snr_L1"] = lvk_result.snr_L1
        gw_prms["snr_V1"] = lvk_result.snr_V1
        gw_prms["detector_available"] = lvk_result.detector_available

        gw_prms["effective_luminosity_distance"] = (
            lensed.effective_luminosity_distance
        )
        gw_prms["effective_geocent_time"] = lensed.effective_geocent_time
        gw_prms["effective_phase"] = lensed.effective_phase
        gw_prms["effective_ra"] = lensed.effective_ra
        gw_prms["effective_dec"] = lensed.effective_dec
        gw_prms["magnifications"] = lensed.magnifications
        gw_prms["time_delays"] = lensed.time_delays
        gw_prms["x_image"] = lensed.x_image
        gw_prms["y_image"] = lensed.y_image

        em_prms = {
            "snr_lens": euclid_result.snr_lens,
            "snr_source": euclid_result.snr_source,
            "magnification": euclid_result.magnification,
            "score": euclid_result.score,
            "resolved": euclid_result.resolved,
            "source_size_ok": euclid_result.source_size_ok,
            "score_ok": euclid_result.score_ok,
            "source_snr_ok": euclid_result.source_snr_ok,
            "observable": euclid_result.observable,
        }

        data = {
            "source_prms": convert_dict(source.to_dict()),
            "lens_prms": convert_dict(lens.to_dict()),
            "gw_prms": convert_dict(gw_prms),
            "em_prms": convert_dict(em_prms),
            "timings": timings,
        }

        os.makedirs(job_out_dir, exist_ok=True)
        filename = os.path.join(
            job_out_dir, f"System_{n_det}_job{job_id}.yaml"
        )

        with open(filename, "w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    if n_det < n_systems:
        print(f"\nStopped: reached max_system_attempts = {max_system_attempts}")
        print(f"Detected {n_det}/{n_systems} systems")
    else:
        print("\nAll systems successfully generated!")

    t_code = time.perf_counter() - tstart
    print(f"Total code execution time: {t_code}")

    return n_det, system_attempts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Simulate joint EM (Euclid) + GW (LVK) detection of "
            "quadruply-imaged lensed BBH mergers using the mml package."
        )
    )

    parser.add_argument(
        "job_id",
        type=int,
        help="Job/task index (also used as an RNG seed offset).",
    )
    parser.add_argument(
        "--num-gw",
        type=int,
        default=4,
        help="Number of lensed GW images required for detection (default: 4).",
    )
    parser.add_argument(
        "--n-systems",
        type=int,
        default=100,
        help="Number of detected systems to generate (default: 100).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="lens_catalog/new_quad_im",
        help="Base output directory (default: lens_catalog/quad_im).",
    )
    parser.add_argument(
        "--max-system-attempts",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--max-lens-attempts",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--max-gw-attempts",
        type=int,
        default=2_000,
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    run(
        job_id=args.job_id,
        num_gw=args.num_gw,
        n_systems=args.n_systems,
        out_dir=args.out_dir,
        max_system_attempts=args.max_system_attempts,
        max_lens_attempts=args.max_lens_attempts,
        max_gw_attempts=args.max_gw_attempts,
    )