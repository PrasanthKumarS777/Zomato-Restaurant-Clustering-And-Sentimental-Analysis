# Zomato-Restaurant-Clustering-And-Sentimental-Analysis

Data science thrives on analytical problem-solving and sharp critical thinking. Among machine learning techniques, unsupervised learning stands out, powering key tasks like clustering, association rules, and dimensionality reduction.

This project tackles a real-world challenge: analyzing Zomato restaurant data across Indian cities to group establishments into meaningful clusters, while also diving into customer reviews for sentiment insights—uncovering whether diners felt positively or negatively about their experiences.

In today's app-driven food scene, platforms like Zomato don't just deliver meals; they create vibrant spaces for sharing honest feedback on restaurants and cafes.

Step 1: Defining the Problem
The core objective? Segment Zomato restaurants by city based on factors like cuisines and average dining costs, then gauge review sentiments to reveal customer vibes.

Step 2: Getting to Know the Data
I started by loading two key datasets: one with comprehensive restaurant details (cuisines, costs, etc.) and another capturing user ratings and reviews. Quick checks with .head() and .info() gave me a solid first look.

Step 3: Data Wrangling
Next, I cleaned things up—converting cost fields to integers, swapping odd ratings with means, and fixing inconsistencies to make the data reliable.

Step 4: Exploratory Data Analysis (EDA)
With clean data in hand, EDA unlocked hidden patterns. Visualizations revealed key relationships, like cost trends and cuisine popularity, yielding actionable insights.

Step 5: Hypothesis Testing
I tested assumptions about the data (e.g., cost vs. ratings), using p-values against a significance threshold to confirm or debunk them.

Step 6: Feature Engineering
Prepped the data for modeling by tackling nulls, outliers, scaling features, and extracting/ selecting the most relevant ones for robust performance.

Step 7: Clustering on Restaurant Data
Applied PCA for dimensionality reduction first, then tested three clustering methods:

K-Means

Agglomerative Hierarchical Clustering

DBSCAN

Optimal results showed 6 natural clusters, validated by Silhouette scores favoring K-Means and Hierarchical approaches.

Step 8: Sentiment Analysis on Reviews
For the review dataset, I preprocessed text rigorously: lowercasing, tokenization via TF-IDF, punctuation/emoji removal, stopword filtering, and lemmatization.

Tested these classifiers:

Logistic Regression

Decision Tree

Random Forest

XGBoost

KNN

Logistic Regression topped the charts with the highest AUC-ROC score. Hyperparameter tuning confirmed it's deployment-ready for sentiment predictions.


# **Data Description**

## Zomato Restaurant names and Metadata

Name : Name of Restaurants

Links : URL Links of Restaurants

Cost : Per person estimated Cost of dining

Collection : Tagging of Restaurants w.r.t. Zomato categories

Cuisines : Cuisines served by Restaurants

Timings : Restaurant Timings

##Zomato Restaurant reviews

Restaurant : Name of the Restaurant

Reviewer : Name of the Reviewer

Review : Review Text

Rating : Rating Provided by Reviewer

MetaData : Reviewer Metadata - No. of Reviews and followers

Time: Date and Time of Review

Pictures : No. of pictures posted with review






**Zomato is an Indian restaurant aggregator and food delivery start-up founded by Deepinder Goyal and Pankaj Chaddah in 2008. Zomato provides information, menus and user-reviews of restaurants, and also has food delivery options from partner restaurants in select cities.

India is quite famous for its diverse multi cuisine available in a large number of restaurants and hotel resorts, which is reminiscent of unity in diversity. Restaurant business in India is always evolving. More Indians are warming up to the idea of eating restaurant food whether by dining outside or getting food delivered. The growing number of restaurants in every state of India has been a motivation to inspect the data to get some insights, interesting facts and figures about the Indian food industry in each city. So, this project focuses on analysing the Zomato restaurant data for each city in India.

The Project focuses on Customers and Company, you have to analyze the sentiments of the reviews given by the customer in the data and made some useful conclusion in the form of Visualizations. Also, cluster the zomato restaurants into different segments. The data is vizualized as it becomes easy to analyse data at instant. The Analysis also solve some of the business cases that can directly help the customers finding the Best restaurant in their locality and for the company to grow up and work on the fields they are currently lagging in.

This could help in clustering the restaurants into segments. Also the data has valuable information around cuisine and costing which can be used in cost vs. benefit analysis

Data could be used for sentiment analysis. Also the metadata of reviewers can be used for identifying the critics in the industry.**
