===========================================================================
# A Comparative Analysis of Amazon Book Ratings Using Collaborative Filtering

Constructing Recommender Models by analyzing the Amazon Book Ratings data and juxtaposing different algorithms based on 
the concept of Collaborative Filtering to perceive the ideal approach. This research is a work in progress and is an expansion of the [Exploratory Data Analysis of Amazon Books Reviews](https://github.com/shahriar-rahman/EDA-Amazon-Books-Reviews)

<br/>

![alt text](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/main/img/amazon%20(5).jpg)

<br/>

### ◘ Introduction
People depend on recommender systems on a regular basis, whether for news updates, stock markets, traveling guides, 
spoken words, advertisements, reference letters, global surveys, and so forth. The natural social procedure is heavily expanded 
upon by the aforementioned systems by assisting in sifting through massive chunks of data related to others which can be 
attributed to websites, movies, music, arts, articles, jokes, finance, and books. Therefore, it is no surprise that algorithms 
such as Collaborative Filtering (CF) have been widely adopted as one of the crucial pieces of the large puzzle that is the 
recommenders. 

<br/><br/>

### ◘ Objective
The primary incentive of this research is to: 
* Initiate an exploratory analysis of the acquired data to find indispensable patterns of the feature that makes up the data.
* Conduct a comparative analysis of features.
* Utilize multiple types of CF on the processed features.
* Experiment with different Hyper-parameters to obtain a well-organized tuning for the models.
* Compare and analyze which models display the most robust generalization.

<br/>

### ◘ Approach
This research is classified into 6 steps:
1.	Identifying the problem and its data sources.
2.	Construct the raw data into clean processed data and explore it using both Jupyter Notebooks and PyCharm IDE.
3.	Make various modifications for safeguarding convenience for the Machine Learning (ML) Model to process.
4. Apply Non-negative Matrix Factorization (NMF), Singular Value Decomposition (SVD), and K-Means to train and evaluate the models using test data.
5.	Experiment and Diagnose in order to achieve the best Hyper-parameters for building efficient models.
6.	Result Analysis and Comparison.

<br/>

![alt text](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/main/img/amazon%20(14).jpg)

<br/>

### ◘ Methodologies & Technologies applied
* Diagnose and fix structural errors
* Check and Clean data
* Address duplicates & perform de-duplication
* Maintain feature consistency
* Deep Feature Exploration	
* Construct SVD, NMF & K-Means models
* Train & Evaluate the generalization of the models
* Apply test set to the previously trained models
* Compare & Analyze the results

<br/>

### ◘ Required Modules
* pandas 2.0.0
* missingNo 0.5.2
* matplotlib 3.7.1
* seaborn 0.12.2
* scikit-learn   1.2.2
* scikit-surprise 1.1.3
* plotly 5.15.0
* numpy 1.24.2

<br/><br/>

### ◘ Project Organization
------------
    ├─-- LICENSE
    |
    ├─-- README.md          <- The top-level README for developers using this project.
    |
    ├─-- data         		<- The original, immutable data dump.
    |
    |
    ├─-- models             <- Trained and serialized models, model predictions, or model summaries  
    |    └── model.pkl
    |
    ├─-- notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    |                         		initials, and a short `-` delimited description
    |
    ├─-- references         <- Data dictionaries, manuals, and all other explanatory materials.
    |
    ├─ figures            <- Generated graphics and figures obtained from visualization.py
    |   └── ide_graphs           <- Generated using PyCharm IDE
    |   |
    |   └── notebook_graphs    <- Generated using Jupyter Notebooks
    |
    ├─-- img            <- Project related files
    |
    ├─-- requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    |                         generated with `pip freeze > requirements.txt`
    |
    ├─-- setup.py           <- makes project pip installable, so that src can be imported
    |
    |
    ├─-- src                <- Source code for use in this project.
    |   └───-- __init__.py    
    |   |
    |   ├─-- data           <- Scripts to download or generate data
    |   |   └── make_dataset.py
    |   |
    |   ├─-- features       <- Scripts to turn raw data into features for modeling
    |   |   └── construct_features.py
    |   |   └──  feature_analysis.py
    |   |   └── parent_features.py
    |   |
    |   ├─-- models         <- Scripts to train models and then use trained models to make predictions         
    |   |   └─── model_test.py
    |   |   └─── model_train.py
    |   |
    |   └───-- visualization  <- Scripts to create exploratory and results oriented visualizations
    |       └───-- visualize.py
    |
    ├─-- tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
--------
<br/><br/>
===========================================================================
