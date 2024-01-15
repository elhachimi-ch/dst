import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
    name="data-science-toolkit",
    version="0.1.063",
    author="EL HACHIMI CHOUAIB",
    author_email="elhachimi.ch@gmail.com",
    description="Data Science Toolkit (DST) is a Python library that helps implement data science related project with ease.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elhachimi-ch/dst",
    project_urls={
        "Bug Tracker": "https://github.com/elhachimi-ch/dst/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.5",
    install_requires=[
        "setuptools>=42",
        "wheel",
        "pandas",
        "numpy==1.25",
        "tensorflow",
        "scikit-learn",
        "seaborn",
        "matplotlib",
        "wordcloud",
        "keras",
        "plotly",
        "xgboost",
        "opencv-python",
        "scikit-image",
        "unidecode",
        "emoji>=1.7",
        "textblob",
        "nltk",
        "wordcloud",
        "optuna",
        "langdetect"
        "xarray",
        "openpyxl",
        "catboost"
    ],
)
