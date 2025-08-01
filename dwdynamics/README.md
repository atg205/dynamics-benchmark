Tools for simulating dynamics of quantum systems using QUBO formalism.

Based on:
```text
Jałowiecki, K., Więckowski, A., Gawron, P. et al. Parallel in time dynamics with quantum annealers. Sci Rep 10, 13534 (2020). https://doi.org/10.1038/s41598-020-70017-x
```

## What is it?

This repo contains a library called `dwdynamics` with tools for generating QUBOs for simulating dynamics of quantum systems using D-Wave machines, or, more generally, using any `dimod`-based sampler. It also contains utilities for interpreting solutions of such QUBOs as state vectors.

## How to install and use

As always, using a fresh virtualenv is recommended. To install `dwdynamics` you can use your favourite method of installation, e.g. clone it and then run `pip install .`.

If you want to run notebooks, include `notebooks` extras, e.g. `pip install ".[notebooks]"`. Including `notebooks` extras will also install `jupyter` and `matplotlib`.

If you plan to develop the package, use poetry, i.e.

```bash
pip install poetry
poetry install -E notebooks
```

For example usage see `notebooks` directory, where we reproduce part of the experiment presented in our original paper. The API is pretty simplistic and the examples are well-commented.
