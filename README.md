# Data Science Toolkit (DST)

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://data-science-toolkit.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Reproducible Capsule](https://img.shields.io/static/v1?label=&message=code+ocean&color=blue)](https://codeocean.com/capsule/1309232/tree)

Data Science Toolkit (DST) is a Python library that helps implement data science projects with ease: from data ingestion and preprocessing to modeling, geospatial analysis, computer vision, text vectorization, and reinforcement learning.

It bundles practical, production-friendly utilities and higher-level abstractions so you can move faster while keeping control over the details.

## Key Features

- **Data handling**: `DataFrame` for loading CSV/JSON/Excel/Parquet, cleaning, transforming, and streaming large datasets.
- **Modeling**: `Model` for traditional ML and deep learning training, cross-validation, metrics, and GPU helpers.
- **Text & NLP**: `Vectorizer` for bag-of-words/TF-IDF, tokenization, cosine similarity, and projections.
- **Charts**: `Chart` utilities for quick exploratory visuals with Matplotlib/Seaborn/Plotly.
- **GIS**: `GIS` for geospatial data layers, joins, CRS transforms, area/perimeter, and exports.
- **Computer Vision**: `ImageFactory` for resizing, cropping, contour detection, blending, and basic filters.
- **Reinforcement Learning**: `Environment` and `R3` tools to explore policies and custom environments.
- **Crop Simulation**: `CSM` modules for crop water requirement, ET simulations, and monitoring pipelines.
- **Utilities**: `Lib` with climate, math, text processing, IO helpers, and more.

## Installation

DST is published as `data-science-toolkit`.

```bash
pip install data-science-toolkit
```

If you’re installing from source (for development):

```bash
git clone https://github.com/elhachimi-ch/dst.git
cd dst
pip install -e .
```

Notes:
- Requires Python 3.5+.
- Some features (e.g., deep learning, GIS, CV) pull heavier dependencies (TensorFlow, CatBoost, OpenCV, Geo stack). Install times may vary.

## Quickstart

```python
from data_science_toolkit.dataframe import DataFrame
from data_science_toolkit.model import Model

# Load a toy dataset
data = DataFrame()
data.load_dataset('iris')
y = data.get_column('target')
data.drop_column('target')

# Fit a decision tree
model = Model(data_x=data.get_dataframe(), data_y=y, model_type='dt', training_percent=0.8)
model.train()
model.report()          # classification metrics
model.cross_validation(5)
```

### Work with Parquet (large data)

```python
from data_science_toolkit.dataframe import DataFrame

# Stream a Parquet dataset efficiently
df = DataFrame(data_path="path/to/parquet/dir", data_type="parquet", n_workers="auto")
summary = df.describe()  # computes per-column stats without loading entire data into RAM
print(summary)
```

### Text Vectorization

```python
from data_science_toolkit.vectorizer import Vectorizer

documents = [
	"data science is fun",
	"toolkits help data workflows",
	"science advances with good tools"
]

vec = Vectorizer(documents_as_list=documents, vectorizer_type='tfidf', ngram_tuple=(1,2))
matrix = vec.get_matrix()
features = vec.get_features_names()
print(len(features), features[:10])
```

### Geospatial Utilities

```python
from data_science_toolkit.gis import GIS

gis = GIS()
gis.add_data_layer("parcels", "data/parcels.geojson", data_type="sf")
gis.add_area_column("parcels", unit="ha")
gis.to_crs("parcels", epsg="3857")
gis.export("parcels", "out/parcels_3857", file_format="geojson")
```

### Computer Vision Helpers

```python
from data_science_toolkit.imagefactory import ImageFactory

img = ImageFactory("data/sample.jpg")
img.to_gray_scale()
img.gaussian_blur((5,5))
img.save("out/processed.jpg")
```

## Documentation

Full API docs and tutorials live at: https://data-science-toolkit.readthedocs.io

## Contributing

Contributions and suggestions are welcome via GitHub pull requests.

Typical workflow:
- Fork the repo and create a feature branch.
- Install dev dependencies: `pip install -e .`.
- Add tests or notebook snippets where relevant.
- Open a PR with a clear description and examples.

## Maintainership

We’re actively enhancing the repo with new algorithms and utilities. Feedback on priorities is appreciated.

## License

MIT License. See the LICENSE file for details.

## Citation

If you use DST in academic work, please cite the repository and (optionally) reference the Code Ocean capsule for reproducibility: https://codeocean.com/capsule/1309232/tree

Additionally, please cite the following paper:

El Hachimi, Chouaib; Belaqziz, Salwa; Khabba, Saïd; Chehbouni, Abdelghani. 2022. "Data Science Toolkit: An All-in-One Python Library to Help Researchers and Practitioners in Implementing Data Science-Related Algorithms with Less Effort." Software Impacts 12:100240. https://doi.org/10.1016/J.SIMPA.2022.100240

BibTeX (optional):

```bibtex
@article{ElHachimi2022,
   author = {Chouaib El Hachimi and Salwa Belaqziz and Saïd Khabba and Abdelghani Chehbouni},
   doi = {10.1016/J.SIMPA.2022.100240},
   issn = {2665-9638},
   journal = {Software Impacts},
   month = {5},
   pages = {100240},
   publisher = {Elsevier},
   title = {Data Science Toolkit: An all-in-one python library to help researchers and practitioners in implementing data science-related algorithms with less effort},
   volume = {12},
   url = {https://linkinghub.elsevier.com/retrieve/pii/S2665963822000124},
   year = {2022}
}

```

