"""
simulate_lens_systems.py
========================

Simulate strongly lensed BBH mergers that are *jointly* detectable by

    * LVK (H1+L1+V1, with optional duty cycles), and
    * Euclid VIS (electromagnetic host-galaxy light).

Two independent choices control which systems end up in the catalog:

``--sampling-region {auto,double,triple,quad}``   (lensing GEOMETRY)
    The source-plane region the BBH is drawn from. The regions are nested
    "at least k images" regions: quad (>=4) ⊂ triple (>=3) ⊂ double (>=2).
    A BBH drawn from region(k) produces *at least* k lensed images; it may
    produce more (a double-region draw lands in the quad region with
    probability A_quad / A_double). The area of this region is the lensing
    cross-section stored as ``caustic_area`` / ``log_p_cross_sec``.

``--min-detected-images K``   (DETECTABILITY; alias ``--num-gw``)
    The system is kept only if at least K of its lensed images pass the LVK
    detection criteria (``--exact-detected`` makes it exactly K).

With ``--sampling-region auto`` (default) the region is chosen from K
(K=2 -> double, K=3 -> triple, K>=4 -> quad). Because you cannot detect more
images than exist, region(K) contains *every* system with >=K detectable
images, so the catalog is complete and the cross-section is A_region. An
explicit region larger than K's region is still complete (just less
efficient); a region *smaller* than K's (e.g. quad lenses with K=2) yields a
sub-population (``sample_is_complete: false`` in the YAML).

Usage
-----
    python simulate_lens_systems.py <job_id> --min-detected-images 2
    python simulate_lens_systems.py <job_id> --min-detected-images 2 \
        --sampling-region quad                 # quads with >=2 detectable
    python simulate_lens_systems.py <job_id> --min-detected-images 3 --exact-detected
    python simulate_lens_systems.py <job_id> --min-detected-images 2 \
        --duty-cycles 0.7 0.7 0.75 --min-detectors 2

``--exact-detected``: by default K means "at least K detectable images". With
this flag the system is kept only if *exactly* K images are detectable (e.g.
K=2 rejects systems where a third image is also detectable). The sampling
region and cross-section are unchanged; only the acceptance test differs. If
you want several multiplicities, it is usually better to run once with
``--min-detected-images 2`` (no flag) and bin on ``n_images_detected`` in the
YAML afterwards.

Duty cycles: ``--duty-cycles H1 L1 V1`` accepts three probabilities (0-1)
in the fixed detector order H1, L1, V1. They are drawn independently for
every image and detector. Default: ``1.0 1.0 1.0`` (always online). Note that
``--min-detectors`` counts only *operating* detectors, so with duty cycles
below 1 the default of 3 requires all three to be online for every image;
``--min-detectors 2`` is usually what you want.

Output: ``<out-dir>/job_<job_id>/System_<n>_job<job_id>.yaml``.
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
from mml.lensing.caustics import region_name
from mml.simulation.sampling import resolve_sampling
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
    min_detected_images: int = 2,
    sampling_region="auto",
    exact_detected: bool = False,
    n_systems: int = 100,
    out_dir: str | None = None,
    max_system_attempts: int = 100_000,
    max_lens_attempts: int = 1_000,
    max_gw_attempts: int = 2_000,
    theta_ein_min: float | None = None,
    min_detectors: int = 3,
    duty_cycles: list[float] | None = None,
    min_abs_magnification: float = 0.1,
    seed_offset: int = 5678,
):
    """
    Simulate ``n_systems`` jointly EM+GW detected lensed-BBH systems.

    Parameters
    ----------
    job_id : int
        Job/task index. Seeds the RNG and namespaces the output files.

    min_detected_images : int
        Number of lensed images that must be LVK-detectable (>= 2).
        This is a *detection* requirement, not a lensing-geometry one.

    sampling_region : {"auto", "double", "triple", "quad"}
        Source-plane region the BBH is drawn from (lensing geometry).
        See the module docstring. "auto" -> smallest region that is
        complete for ``min_detected_images``.

    exact_detected : bool
        Require exactly ``min_detected_images`` detectable images instead of
        at least that many.

    out_dir : str, optional
        Base output directory. Defaults to a name encoding the region and
        the detection requirement, so different settings never mix.

    duty_cycles : list of float, optional
        Three duty-cycle probabilities in H1, L1, V1 order. Each must lie
        between 0 and 1. Defaults to [1.0, 1.0, 1.0].

    min_abs_magnification : float
        |mu| threshold used only to count "bright" images for the catalog
        (matches the |mu| = 0.1 cut baked into the double region).

    Other parameters are retry budgets, detector settings and the seed offset.
    """

    if theta_ein_min is None:
        theta_ein_min = EUCLID.theta_ein_threshold

    region_min, complete = resolve_sampling(
        min_detected_images, sampling_region
    )
    region_label = region_name(region_min)

    if out_dir is None:
        out_dir = os.path.join(
            "lens_catalog",
            f"{region_label}_region_"
            f"{'eq' if exact_detected else 'ge'}{min_detected_images}det",
        )

    print(
        f"Sampling BBH positions in the '{region_label}' region "
        f"(>= {region_min} images); requiring "
        f"{'exactly' if exact_detected else 'at least'} "
        f"{min_detected_images} detectable images. "
        f"Complete sample: {complete}."
    )

    rng = np.random.default_rng(seed_offset + job_id)

    lens_sampler = Lens(sigmazfn="cond_on_zs", rng=rng)
    source_sampler = Source(rng=rng)
    position_sampler = BBHPositionSampler(rng=rng)
    gw_population = GWPopulation(rng=rng)

    # Duty cycles are supplied as a three-element list in H1, L1, V1
    # order. LVKDetector currently accepts a detector-name mapping, so
    # convert the public list interface to that internal representation.
    if duty_cycles is None:
        duty_cycles = [1.0, 1.0, 1.0]

    if len(duty_cycles) != 3:
        raise ValueError(
            "duty_cycles must contain exactly three values in H1, L1, V1 order."
        )

    duty_cycle_map = dict(zip(("H1", "L1", "V1"), duty_cycles))

    # The detector's RNG is seeded from the simulation RNG so that
    # duty-cycle draws are reproducible with job_id.
    lvk_detector = LVKDetector(
        num_detected_images=min_detected_images,
        exact_num_images=exact_detected,
        min_detectors=min_detectors,
        duty_cycles=duty_cycle_map,
        random_seed=int(rng.integers(2**31 - 1)),
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
        # BBH + HOST POSITION: uniform inside the chosen region
        # (lensing geometry: >= region_min images are produced)
        # ------------------------------------------------------------
        try:
            position = position_sampler.sample(
                kwargs_lens=kwargs_lens,
                kwargs_source=kwargs_source,
                min_lensed_images=region_min,
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            print(f"Error while sampling BBH/source positions: {e}")
            continue

        # ------------------------------------------------------------
        # SOLVE THE LENS EQUATION ONCE. Images, magnifications, delays and
        # Morse phases do not depend on the BBH parameters, so the (up to
        # 2000) GW draws below re-use this solution.
        # ------------------------------------------------------------
        try:
            solution = gw_population.solve_images(
                lens=lens,
                x_gw=position.x_gw,
                y_gw=position.y_gw,
                z_source=source.z_source,
            )
        except RuntimeError as e:
            print(f"Lens equation solver failed: {e}")
            continue

        # Cannot detect more images than exist: skip hopeless geometries
        # instead of burning max_gw_attempts on them.
        if solution.n_images < min_detected_images:
            continue

        # ------------------------------------------------------------
        # EM (EUCLID) DETECTION
        # ------------------------------------------------------------
        em_start = time.perf_counter()

        euclid_result = calculate_euclid_snr(lens, source, position)

        timings["em_detection_time"] = time.perf_counter() - em_start
        timings["em_attempts"] = 1

        if not euclid_result.observable:
            continue

        # ------------------------------------------------------------
        # GW SAMPLING + LVK DETECTION
        # (retry until the detection requirement is met)
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

            lensed = gw_population.apply_lensing(
                gw=gw,
                solution=solution,
                x_gw=position.x_gw,
                y_gw=position.y_gw,
                z_source=source.z_source,
            )

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

        # --- selection / cross-section bookkeeping -------------------
        # caustic_area is the area of the region the BBH was drawn from,
        # i.e. the cross-section for "produces >= region_min_images".
        gw_prms["sampling_region"] = region_label
        gw_prms["region_min_images"] = region_min
        gw_prms["min_detected_images_required"] = min_detected_images
        gw_prms["exact_detected_required"] = exact_detected
        gw_prms["sample_is_complete"] = complete
        gw_prms["duty_cycles"] = list(duty_cycles)
        gw_prms["min_detectors"] = min_detectors
        gw_prms["caustic_area"] = position.caustic_area
        gw_prms["log_p_cross_sec"] = log_cross_section(position.caustic_area)

        # --- images PRODUCED by the geometry vs images DETECTED ------
        gw_prms["n_images_produced"] = solution.n_images
        gw_prms["n_images_bright"] = solution.n_images_above(
            min_abs_magnification
        )
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


def _probability(text):
    """argparse type: float in [0, 1]."""
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"duty cycle must be between 0 and 1, got {value}"
        )
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Simulate joint EM (Euclid) + GW (LVK) detection of strongly "
            "lensed BBH mergers using the mml package."
        )
    )

    parser.add_argument(
        "job_id",
        type=int,
        help="Job/task index (also used as an RNG seed offset).",
    )
    parser.add_argument(
        "--min-detected-images",
        "--num-gw",  # deprecated alias of the old single option
        dest="min_detected_images",
        type=int,
        default=2,
        help=(
            "Number of lensed GW images that must be detectable "
            "(default: 2). Detection requirement only; does not by itself "
            "fix the lensing geometry (see --sampling-region)."
        ),
    )
    parser.add_argument(
        "--sampling-region",
        choices=["auto", "double", "triple", "quad"],
        default="auto",
        help=(
            "Source-plane region the BBH is drawn from: double (>=2 "
            "images), triple (>=3), quad (>=4). 'auto' (default) picks the "
            "smallest region that is complete for --min-detected-images."
        ),
    )
    parser.add_argument(
        "--exact-detected",
        action="store_true",
        help="Require exactly (not at least) --min-detected-images detected.",
    )
    parser.add_argument(
        "--min-detectors",
        type=int,
        default=3,
        help="Operating detectors that must each exceed the per-detector "
        "SNR threshold for an image to count (default: 3).",
    )
    parser.add_argument(
        "--duty-cycles",
        nargs=3,
        type=_probability,
        metavar=("H1", "L1", "V1"),
        default=[1.0, 1.0, 1.0],
        help=(
            "Three detector duty cycles (probabilities between 0 and 1), "
            "in H1 L1 V1 order. Default: 1.0 1.0 1.0. "
            "Each detector's availability is sampled independently "
            "for every image."
        ),
    )
    parser.add_argument("--n-systems", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base output directory (default: encodes region and detection "
        "requirement, e.g. lens_catalog/double_region_ge2det).",
    )
    parser.add_argument("--max-system-attempts", type=int, default=100_000)
    parser.add_argument("--max-lens-attempts", type=int, default=1_000)
    parser.add_argument("--max-gw-attempts", type=int, default=2_000)

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    run(
        job_id=args.job_id,
        min_detected_images=args.min_detected_images,
        sampling_region=args.sampling_region,
        exact_detected=args.exact_detected,
        n_systems=args.n_systems,
        out_dir=args.out_dir,
        max_system_attempts=args.max_system_attempts,
        max_lens_attempts=args.max_lens_attempts,
        max_gw_attempts=args.max_gw_attempts,
        min_detectors=args.min_detectors,
        duty_cycles=args.duty_cycles,
    )