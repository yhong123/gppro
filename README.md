# Product-of-experts Gaussian Process Model

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/yhong123/gppro/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/yhong123/gppro/actions/workflows/tests.yml
[linting-badge]:            https://github.com/yhong123/gppro/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/yhong123/gppro/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/yhong123/gppro/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/yhong123/gppro/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A python package implementing a product-of-experts Gaussian process model, which consists of a collection of local GP models that work collaboratively, and incorporates an information-based method to calibrate the overestimated posterior variances.

See paper [Information-based Calibration of Uncertainty Quantification in Product of Gaussian Process Models](https://github.com/yhong123/doc/main/paper_1.pdf).

<!-- This project is developed in collaboration with the  -->
<!-- [Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.  -->

<!-- ## About  -->

<!-- ### Project Team  -->

<!-- YH Ong ([yean-hoon.ong@ucl.ac.uk](mailto:yean-hoon.ong@ucl.ac.uk))  -->

<!-- TODO: how do we have an array of collaborators ? -->

<!-- ### Research Software Engineering Contact  -->

<!-- Centre for Advanced Research Computing, University College London  -->
<!-- ([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))  -->


## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`gppro` requires Python 3.9&ndash;3.12.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using
a environment management tool such as
[Conda](https://docs.conda.io/projects/conda/en/stable/). To install the latest
development version of `gppro` using `pip` in the currently active
environment run

```sh
pip install git+https://github.com/yhong123/gppro.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/yhong123/gppro.git
```

and then install in editable mode by running

```sh
pip install -e .
```

