# MML Joint Detection

A Python package for simulating strongly lensed gravitational-wave (GW) and
electromagnetic (EM) observations of their strongly lensed host galaxy counterparts,
with an emphasis on joint LVK and Euclid detectability.

The package is organised into reusable modules for source and lens populations,
lensing calculations, detector response, simulation utilities, and output I/O.

## Installation

Clone the repository and create/activate an environment containing the project
dependencies. From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

For development and tests, install the development extras if they are defined
in `pyproject.toml`, then run:

```bash
python -m pytest
```

Editable installation means changes under `src/mml/` are available immediately
without setting `PYTHONPATH`.

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


## Tests

Run the test suite from the repository root:

```bash
python -m pytest -v
```

The package uses a `src/` layout. Import modules through `mml`, rather than
running files inside `src/mml/` directly as standalone scripts.