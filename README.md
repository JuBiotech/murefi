[![PyPI version](https://img.shields.io/pypi/v/murefi)](https://pypi.org/project/murefi)
[![pipeline](https://github.com/jubiotech/murefi/workflows/pipeline/badge.svg)](https://github.com/jubiotech/murefi/actions)
[![coverage](https://codecov.io/gh/jubiotech/murefi/branch/master/graph/badge.svg)](https://codecov.io/gh/jubiotech/murefi)
[![documentation](https://readthedocs.org/projects/murefi/badge/?version=latest)](https://murefi.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/353352505.svg)](https://zenodo.org/badge/latestdoi/353352505)

# `murefi`
This package provides useful data structures and mapping objects for __mu__&#x200b;lti-__re__&#x200b;plicate __fi__&#x200b;tting, mainly of ordinary differential euqation (ODE) models.
To see implementation examples & excercises, you can go to [notebooks/](notebooks).

# Installation
`murefi` is released on [PyPI](https://pypi.org/project/murefi/):

```
pip install murefi
```
# Documentation
Read the package documentation [here](https://murefi.readthedocs.io/en/latest/?badge=latest).

# Usage and Citing
`murefi` is licensed under the [GNU Affero General Public License v3.0](https://github.com/jubiotech/murefi/blob/master/LICENSE).

When using `murefi` in your work, please cite the [Helleckes & Osthege et al. (2022) paper](https://doi.org/10.1371/journal.pcbi.1009223) __and__ the [corresponding software version](https://doi.org/10.5281/zenodo.4652911).

Note that the paper is a shared first co-authorship, which can be indicated by <sup>1</sup> in the bibliography.

```bibtex
@article{calibr8Paper,
  doi       = {10.1371/journal.pcbi.1009223},
  author    = {Helleckes$^1$, Laura Marie and
               Osthege$^1$, Michael and
               Wiechert, Wolfgang and
               von Lieres, Eric and
               Oldiges, Marco},
  journal   = {PLOS Computational Biology},
  publisher = {Public Library of Science},
  title     = {Bayesian and calibration, process modeling and uncertainty quantification in biotechnology},
  year      = {2022},
  month     = {03},
  volume    = {18},
  url       = {https://doi.org/10.1371/journal.pcbi.1009223},
  pages     = {1-46},
  number    = {3}
}
@software{murefi,
  author       = {Michael Osthege and
                  Laura Helleckes},
  title        = {JuBiotech/murefi: v5.1.0},
  month        = feb,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v5.1.0},
  doi          = {10.5281/zenodo.6006488},
  url          = {https://doi.org/10.5281/zenodo.6006488}
}
```

Head over to Zenodo to [generate a BibTeX citation](https://doi.org/10.5281/zenodo.4652911) for the latest release.
