# MML Joint Detection

A Python package for simulating strongly lensed gravitational-wave (GW) and
electromagnetic (EM) observations of their strongly lensed host galaxy counterparts,
with an emphasis on joint LVK and Euclid detectability.

The package is organised into reusable modules for source and lens populations,
lensing calculations, detector response, simulation utilities, and output I/O.

## Installation

The following procedure checks that the repository can be cloned, installed in a
clean Python environment, and used to launch the example simulation.

### 1. Clone the repository

```bash
git clone --branch refactor/project-structure \
    https://github.com/disharh/mml-joint-detection.git

cd mml-joint-detection
```

### 2. Create and activate a clean environment

This example uses Python 3.11:

```bash
conda create -n mml-test python=3.11
conda activate mml-test
```

### 3. Install the package

Upgrade `pip` and install the repository in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation makes the `mml` package importable without setting
`PYTHONPATH` manually.

### 4. Check the package import

```bash
python -c "import mml; print(mml.__file__)"
```

The output should point to the `src/mml/` directory in the cloned repository.

### 5. Check source-population model loading

The default source-population sampler loads a trained density-estimation model.
Check that the model is present in the repository at the path expected by
`Source`:

```bash
python -c "from mml.populations import Source; s = Source(); print(s.model_path)"
```

If this command raises `FileNotFoundError`, check that the trained model files
are included in the clone and stored in the expected package data directory.

### 6. Check the example command-line interface

Run the help command to display the available arguments:

```bash
python simulate_lens_systems.py --help
```

### 7. Run a one-system smoke test

Try generating one detected lens system with reduced attempt limits:

```bash
python simulate_lens_systems.py 0 \
    --n-systems 1 \
```
The simulation generates interpolators (for the first time), performs lensing calculations, Euclid image simulations, and
GW signal-to-noise calculations, so even a one-system run may take some time.

## Package layout

```text
src/mml/
├── data/          # Trained models and packaged input data
├── detectors/     # GW and EM detection criteria
├── io/            # Input/output helpers
├── lensing/       # Lensing geometry and image calculations
├── populations/   # Source, lens, and GW population models
├── rates/         # TO BE ADDED: Detection-rate calculations
├── simulation/    # Simulation and sampling orchestration
└── utils/         # Shared utilities, including cache-path handling

examples/           # End-to-end example workflows
tests/              # Automated tests
```

## Source population models

`mml.populations.Source` uses pre-trained `denmarf`/MAF models. By default,
the following files are expected directly inside `src/mml/data/trained_models/`:

- `trained_de_weighted_mass_b32_h128_e1000_sizen.pkl`
- `trained_de_b32_h128_e1000_bdef_wbounds.pkl`

Select a model with:

```python
from mml.populations import Source

source_sampler = Source(model="non_weighted")
sources = source_sampler.sample(size=100)
```

Use `model="mstar_weighted"` for the stellar-mass-weighted model. To load a
model from another location, pass `model_path="/path/to/model.pkl"`.

## Lens redshift and velocity-dispersion sampling

The conditional sampler is available as:

```python
from mml.populations.conditional_sigma_z import sample_sigma_zl_given_zs

sigma_lens, z_lens = sample_sigma_zl_given_zs(
    z_source=[0.8, 1.2, 2.0],
)
```

The precomputed normalisation grid `M(z)` is generated on first use and cached
outside the repository, by default under:

```text
~/.cache/mml/lens_population/
```

Pass `cache_file=...` to override the cache location explicitly.