# PyGitHub

[![PyPI](https://img.shields.io/pypi/v/PyGithub.svg)](https://pypi.python.org/pypi/PyGithub)
![CI](https://github.com/PyGithub/PyGithub/workflows/CI/badge.svg)
[![readthedocs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://data-science-toolkit.ml/)
[![License](https://img.shields.io/badge/license-LGPL-blue.svg)](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License)
[![Slack](https://img.shields.io/badge/Slack%20channel-%20%20-blue.svg)](https://join.slack.com/t/pygithub-project/shared_invite/zt-duj89xtx-uKFZtgAg209o6Vweqm8xeQ)
[![Open Source Helpers](https://www.codetriage.com/pygithub/pygithub/badges/users.svg)](https://www.codetriage.com/pygithub/pygithub)
[![codecov](https://codecov.io/gh/PyGithub/PyGithub/branch/master/graph/badge.svg)](https://codecov.io/gh/PyGithub/PyGithub)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyGitHub is a Python library help implement data science related project with ease.

[GitHub REST API]: https://docs.github.com/en/rest
[GitHub]: https://github.com

## Simple Demo

```python
data = DataFrame()
data.load_dataset('iris')
y = data.get_column('target')
data.drop_column('target')

# decision tree model
model = Model(data_x=data.get_dataframe(), data_y=y, model_type='dt', training_percent=0.8)

# train the model
model.train()

# get all classification evaluation metrics
model.report()

#get the cross validation
model.cross_validation(5)
```


## Documentation

More information can be found on the [PyGitHub documentation site.](https://pygithub.readthedocs.io/en/latest/introduction.html)

### Contributing

Contrubution and suggestions are welcome via GitHub Pull Requests.

### Maintainership

We're actively enhacing the repo with new algorithms.
