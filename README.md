Readme
==============================

Predicting housing prices on Kaggle. Not meant to be a serious attempt at scoring a gold medal on the leaderboard, but rather a fun little demonstration about how I would quickly assemble and structure a project in a few hours.

# Project structure

1) Preprocess the data, primarily through imputation and scaling (src/data/make_dataset.py).

2) Select relevant features using permutation feature importance (src/features/build_features.py).

3) Run through some candidate models (src/models/eval\_models.py), then tune its hyperparameters (src/models/train\_model.py) and make predictions (src/models/predict\_model.py).

4) Output the predictions in data/processed/predictions.csv.


# Running the code

To run the code from the command line, you'll need git, the [Kaggle API](https://www.kaggle.com/docs/api) for getting the data, and [conda](https://conda.io).

```bash
# clone the repo
git clone https://github.com/sharpwaveripple/house-price-prediction

# get the data using the Kaggle API
kaggle competitions download -c house-prices-advanced-regression-techniques

# move it into data/raw
mv house-prices-advanced-regression-techniques.zip house-price-prediction/data/raw/

# enter the directory and install the conda environment
cd house-price-prediction/
conda env create -f environment.yml
conda activate kaggle-housing

# run the project
cd src/
python main.py
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
