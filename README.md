===========================================================================
# A Comparative Analysis of Amazon Book Ratings using Collaborative Filtering

Constructing Recommender Models by analyzing the Amazon Book Ratings data and juxtaposing different algorithms based on 
the concept of Collaborative Filtering to perceive the ideal approach.

![alt text](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/main/img/amazon%20(5).jpg)

### Introduction:
People depend on recommender systems on a regular basis, whether for news updates, stock markets, traveling guides, 
spoken words, advertisements, reference letters, global surveys, and so forth. The natural social procedure is heavily expanded 
upon by the aforementioned systems by assisting in sifting through massive chunks of data related to others which can be 
attributed to websites, movies, music, arts, articles, jokes, finance, and books. Therefore, it is no surprise that algorithms 
such as Collaborative Filtering (CF) have been widely adopted as one of the crucial pieces of the large puzzle that is the 
recommenders.

### Objective:
The primary incentive of this research is to:
* Initiate an exploratory analysis of the acquired data to find indispensable patterns of the feature that makes up the data.
* Conduct a comparative analysis of features.
* Utilize multiple types of CF on the processed features.
* Experiment with different Hyper-parameters to obtain a well-organized tuning for the models.
* Compare and analyze which models display the most robust generalization.


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
