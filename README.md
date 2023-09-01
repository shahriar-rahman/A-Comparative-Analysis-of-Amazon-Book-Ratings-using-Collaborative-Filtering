===========================================================================
# A Comparative Analysis of Amazon Book Ratings Using Collaborative Filtering

Constructing Recommender Models by analyzing the Amazon Book Ratings data and juxtaposing different algorithms based on 
the concept of Collaborative Filtering to perceive the ideal approach. This research is a work in progress and is an expansion of the [Exploratory Data Analysis of Amazon Books Reviews](https://github.com/shahriar-rahman/EDA-Amazon-Books-Reviews)

<br/>

![alt text](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/main/img/amazon%20(5).jpg)

<br/>

## ◘ Navigation
- [Introduction](#-introduction)
    - [Abstract](#-abstract)
    - [Background](#-background)
    - [Recommender Engine Block Diagram](#-block-diagram-of-a-recommender-engine)
    - [Objectives](#-objectives)
- [Technical Preliminaries](#-technical-preliminaries)
    - [Approach](#-approach)
    - [Dataset](#-datasets)
    - [Book Information](#-book-information)
    - [Book Ratings](#-book-ratings)
    - [Methodologies and Technologies](#-methodologies-and-technologies)
- [Feature Analysis](#-feature-analysis)
    - [Data Procesing](#-data-procesing)
    - [Data Exploration](#-data-exploration)
    - [Feature Analysis Summary Flowchart](#-feature-analysis-summary-flowchart)

<br/><br/>

## ◘ Introduction

### • Abstract
In recent years, Recommendation Systems (RS) have been playing a pivotal role in the world of e-commerce as the utilization of Collaborative Filtering (CF) is getting more popular by the day and has transitioned into an instrumental asset for the RS. This study proposes six different iterations of CF that will act as a base foundation on which many services in the modern world can rely. The models are implemented using three of the most popular algorithms: k-nearest Neighbour (KNN), Singular Value Decomposition (SVD), and Non-negative Matrix Factorization (NMF). Each model is trained using the dataset, acquired from Amazon Book Reviews, and two separate datasets were generated: the first consists of highly processed data of the initial data, and the last consists of hybrid values of the processed data and the Sentimental Analysis of the customer reviews using language processing method. The models evaluated give satisfactory performance with KNN providing the best possible results proving it to be an effective RS tool that can be applied in various industries. 

<br/>

### • Background
People depend on recommender systems on a regular basis, whether for news updates, stock markets, traveling guides, 
spoken words, advertisements, reference letters, global surveys, and so forth. The natural social procedure is heavily expanded 
upon by the aforementioned systems by assisting in sifting through massive chunks of data related to others which can be 
attributed to websites, movies, music, arts, articles, jokes, finance, and books. Therefore, it is no surprise that algorithms 
such as Collaborative Filtering (CF) have been widely adopted as one of the crucial pieces of the large puzzle that is the 
recommenders.   

One of the most important techniques for RS applications is the Collaborative Filtering Process. As stated before, nearly all popular websites like Amazon, YouTube, and Netflix use collaborative filtering as a part of their sophisticated RS. From a hypothetical standpoint, it works by performing a search on a large group of people and observing a smaller set of users with opinions, tastes, and preferences similar to a particular user. It looks at the items they like and amalgamates them to form a ranked list of suggestions. For instance, items for which the content is unavailable could still be recommended to customers through the ratings of other users. Moreover, CF recommendations are based on the quality of the products rated by similar customers instead of being dependent on the content. These methods can recommend items with a diverse range of contents, as long as the neighbors have already shown an interest in these aforementioned items. Research on CF can be classified into two types: instance-based methods and model-based methods.

In this study, several RS have been explored based on CF algorithms to determine the characteristic property of the items based on a user-item relationship. Since Amazon utilizes various types of RS in their services, the primary concept is to implement a base model that can be refurbished and improved upon to be a potential candidate for an effective RS that can be used as an algorithm for Amazon in the future or any other companies that relies on a numerous amount of big data and user-item distinction process. Different popular algorithms are developed to achieve this feat, such as k-nearest Neighbours (KNN), Singular Value Decomposition (SVD), and Non-negative Matrix Factorization (NMF). The dataset was acquired from Amazon and is used to train and evaluate the models’ performance. Primarily, the data that is being fed into the systems alongside users and items is rated which is defined from 1 to 5 stars, with 1 being the lowest interest rate and 5 being the highest. Additionally, robust Language processing is applied to the reviews written by users to generate a hybrid scoring system that amalgamates the initial ratings with the sentiment (i.e., polarity value) of the text. Various pre-processing steps are taken to ensure the data acquired is adequate for model training and evaluation before processing it into the aforementioned algorithms.

<br/>

### • Block Diagram of a Recommender Engine
![alt text](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/branch-updates/figures/Diagrams/block_rs.png)

<br/>

### • Objectives
The primary incentive of this research is to: 
* Initiate an exploratory analysis of the acquired data to find indispensable patterns of the feature that makes up the data.
* Conduct a comparative analysis of features.
* Utilize multiple types of CF on the processed features.
* Experiment with different Hyper-parameters to obtain a well-organized tuning for the models.
* Compare and analyze which models display the most robust generalization.

<br/><br/>

## ◘ Technical Preliminaries

### • Approach
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

### • Datasets
Amazon is an American multinational technology company that focuses primarily on e-commerce, online advertising, cloud computing, digital streaming, and artificial intelligence. Moreover, it has been often referred to as "one of the most influential economic and cultural forces in the world. As a result, the Amazon dataset is chosen for this research in order to establish a benchmark for the recommendation system for any company with historical data.

The model uses Collaborative filtering to evaluate the books' reviews on Amazon. The review data contains the feedback of over 3 million users on unique books and it contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1992 - July 2016. The book detail dataset contains information of all the unique books. The file is built by using Google Book API to get details information about books it rated in the first file. The meta-data of the book attributes of the dataset is provided below.

<br/>

### • Book Information

| Original Attribute | Column Attribute | Data Format | Description |
|--|--|--|--|
| ID | id | object | Book Identifier. |
| Title | title | object | Book Title Name. |
| Price | price | float64 | Price of the Book. |
| User ID | user_id | object | The ID of the user who rates the Book. |
| Profile Name | profile_name | object | Name of the user who rates the Book. |
| Reviews | review/helpfulness | object | Helpfulness rating of the Review |
| Review Scores | review/score | object | Rating from 0 to 5 for the book. |
| Review Time | review/time | object | Time of giving the Review. |
| Review Summary | review/summary | object | The summary of a text Review.|
| Review Text | review/text | object | The full text of a Review. |

<br/>

### • Book Ratings

| Original Attribute | Column Attribute | Data Format | Non-Nulls | Description |
|--|--|--|--|--|
| Title | title | object | 212403 | Book Title. |
| Description | description | object | 143962 | Description of Book. |
| Authors | authors | object | 180991 | Name of book Authors. |
| Image | image | object | 160329 | URL for the Book cover. |
| Preview Link | preview_link | object | 188568 | Link to access Book on Google books. |
| Publisher | publisher | object | 136518 | Name of the Publisher. |
| Published Date | published_date | object | 187099 | The date of Publish. |
| Info Link | info_link | object | 188568 | Link to get more information about the book on Google Books. |
| Categories | categories | object | 171205 | Genres of Books. |
| Ratings Count | ratings_count | float64 | 49752 | The average rating for the Book. |

<br/>

### • Methodologies and Technologies
* Diagnose and fix structural errors
* Check and Clean data
* Address duplicates & perform de-duplication
* Maintain feature consistency
* Deep Feature Exploration	
* Construct SVD, NMF & KNN models
* Train & Evaluate the generalization of the models
* Apply test set to the previously trained models
* Compare & Analyze the results

<br/><br/>

## ◘ Feature Analysis
The dataset contains two files: book ratings and book information. The entire exploratory process is divided into two steps. One of them is data processing and another one is data analysis. The entire codes of the feature analysis as well as data exploration can be accessed [from this link](https://github.com/shahriar-rahman/A-Comparative-Analysis-of-Amazon-Book-Ratings-using-Collaborative-Filtering/blob/branch-updates/notebooks/feature_exploration.ipynb).

<br/>

### • Data Procesing
1. Inquire structural integrity
2. Enhance data accessibility
3. Fix Structural Issues
4. Data cleaning
5. Find duplicates and perform the De-duplication process
6. Validate de-duplication
7. Maintain Feature consistency
8. Review Data frame

<br/>

### • Data Exploration
1. Book Prices and Ratings
2. Density Inspection
3. Pearson Correlation for Numerical features
4. Top 10 Book Genre
5. Book Ratings' effect on its Prices
6. Books most purchased
7. Highest Mean Rated Books
8. Most Expensive Books
9. Top Rated Books accumulating over 3500 ratings
10. Aggregate books for a particular category
11. Authors with the most published books
12. Most Active Years for authors
13. Authors working with multiple genres
14. Review Data frame

<br/>

### • Feature Analysis Summary Flowchart
![alt text](https://github.com/shahriar-rahman/EDA-Amazon-Books-Reviews/blob/main/img/img1.JPG)

<br/><br/>

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
