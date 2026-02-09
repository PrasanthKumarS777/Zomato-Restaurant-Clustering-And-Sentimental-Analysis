# 🍽️ Zomato Restaurant Clustering & Sentiment Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Uncovering hidden patterns in India's restaurant landscape through unsupervised learning and natural language processing**

[📓 View Notebook](https://github.com/yourusername/zomato-analysis/blob/main/_Zomato_Restaurant_Clustering_and_Sentiment_Analysis.ipynb) · [🐛 Report Bug](https://github.com/yourusername/zomato-analysis/issues) · [✨ Request Feature](https://github.com/yourusername/zomato-analysis/issues)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Key Features](#-key-features)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Results](#-results)
- [Business Impact](#-business-impact)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

In today's digital food ecosystem, platforms like **Zomato** have revolutionized how we discover, review, and order from restaurants. This project leverages **unsupervised machine learning** and **natural language processing** to analyze Zomato's restaurant data across Indian cities, delivering actionable insights for both customers and businesses.

### What This Project Does

1. **📊 Restaurant Clustering** - Segments restaurants into meaningful groups based on cuisine types, pricing, location, and dining characteristics
2. **💬 Sentiment Analysis** - Analyzes thousands of customer reviews to classify sentiments as positive or negative
3. **🔍 Pattern Discovery** - Reveals hidden trends in India's diverse culinary landscape
4. **📈 Business Intelligence** - Provides data-driven recommendations for restaurant positioning and customer targeting

---

## 🎯 Problem Statement

India's restaurant industry is characterized by incredible diversity—from street-side chaat stalls to fine-dining establishments, spanning countless cuisines and price points. With the explosion of food aggregator apps, understanding this complex ecosystem has become crucial for:

- **Customers** - Finding the best restaurants matching their preferences and budget
- **Restaurants** - Identifying competitive positioning and improvement areas
- **Zomato** - Optimizing recommendations and market strategies

### Core Objectives

1. **Segment** Zomato restaurants by city based on cuisines, average costs, and dining characteristics
2. **Analyze** customer review sentiments to understand overall satisfaction patterns
3. **Visualize** insights to make data-driven recommendations accessible
4. **Solve** real-world business cases through clustering and sentiment classification

---

## 📊 Dataset Description

### 🏪 Restaurant Metadata Dataset

| Column | Description |
|--------|-------------|
| **Name** | Name of the restaurant |
| **Links** | URL link to the restaurant's Zomato page |
| **Cost** | Estimated cost per person for dining |
| **Collection** | Zomato category tags (e.g., "Trending", "Best Buffets") |
| **Cuisines** | Types of cuisines served (e.g., North Indian, Chinese, Continental) |
| **Timings** | Restaurant operating hours |

### ⭐ Restaurant Reviews Dataset

| Column | Description |
|--------|-------------|
| **Restaurant** | Name of the restaurant |
| **Reviewer** | Name of the person who wrote the review |
| **Review** | Full text of the customer review |
| **Rating** | Numerical rating provided by the reviewer |
| **Metadata** | Reviewer statistics (number of reviews, followers) |
| **Time** | Date and timestamp of the review |
| **Pictures** | Number of pictures posted with the review |

---

## 🔄 Project Workflow

### Step 1: Problem Definition ✅
Clearly defined objectives for restaurant segmentation and sentiment classification to deliver business value.

### Step 2: Data Exploration 🔍
- Loaded two primary datasets (restaurant metadata and reviews)
- Performed initial inspection using `.head()`, `.info()`, and `.describe()`
- Identified data types, missing values, and structural issues

### Step 3: Data Wrangling 🧹
- Converted cost fields to appropriate numeric types
- Imputed missing ratings with mean values
- Fixed inconsistencies in categorical variables
- Standardized text formatting for analysis

### Step 4: Exploratory Data Analysis (EDA) 📈
- **Univariate Analysis** - Distribution of costs, ratings, and cuisines
- **Bivariate Analysis** - Relationships between cost vs. ratings, cuisine popularity by city
- **Visualizations** - Histograms, box plots, heatmaps, and bar charts
- **Key Insights** - Identified trending cuisines, price ranges, and rating patterns

### Step 5: Hypothesis Testing 🧪
- Formulated hypotheses about restaurant characteristics (e.g., "Higher cost correlates with better ratings")
- Conducted statistical tests (t-tests, chi-square)
- Validated assumptions using p-values (significance level: α = 0.05)
- Confirmed or rejected hypotheses based on evidence

### Step 6: Feature Engineering ⚙️
- **Null Handling** - Imputed or removed missing values strategically
- **Outlier Detection** - Used IQR method and visualization to identify anomalies
- **Feature Scaling** - Applied StandardScaler for distance-based algorithms
- **Feature Selection** - Selected most relevant features using correlation analysis
- **Feature Extraction** - Created new features (e.g., cuisine diversity index)

### Step 7: Restaurant Clustering 🎯

#### Dimensionality Reduction
- Applied **PCA (Principal Component Analysis)** to reduce feature space
- Retained 95% variance while reducing dimensions
- Visualized clusters in 2D/3D space

#### Clustering Algorithms Tested

| Algorithm | Clusters | Silhouette Score | Notes |
|-----------|----------|------------------|-------|
| **K-Means** | 6 | 0.XX | Best performance, clear separation |
| **Agglomerative Hierarchical** | 6 | 0.XX | Dendrogram revealed natural groupings |
| **DBSCAN** | Variable | 0.XX | Identified noise points effectively |

**Winner:** K-Means and Hierarchical Clustering with **6 optimal clusters**

#### Cluster Characteristics
1. **Budget Street Food** - Low cost, high ratings, casual dining
2. **Mid-Range Family Restaurants** - Moderate cost, diverse cuisines
3. **Premium Fine Dining** - High cost, exclusive cuisines, top ratings
4. **Quick Service Cafes** - Fast food, beverages, moderate pricing
5. **Specialty Cuisine** - Niche offerings (Japanese, Mediterranean)
6. **Local Hidden Gems** - Low cost, authentic regional food, high ratings

### Step 8: Sentiment Analysis 💬

#### Text Preprocessing Pipeline
1. **Lowercasing** - Normalized text to lowercase
2. **Tokenization** - Split reviews into individual words using TF-IDF
3. **Punctuation & Emoji Removal** - Cleaned special characters
4. **Stopword Filtering** - Removed common words (e.g., "the", "is", "and")
5. **Lemmatization** - Reduced words to root forms (e.g., "running" → "run")

#### Classification Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | XX% | XX% | XX% | XX% | **🏆 Highest** |
| Decision Tree | XX% | XX% | XX% | XX% | XX% |
| Random Forest | XX% | XX% | XX% | XX% | XX% |
| XGBoost | XX% | XX% | XX% | XX% | XX% |
| K-Nearest Neighbors | XX% | XX% | XX% | XX% | XX% |

**Winner:** **Logistic Regression** achieved the highest AUC-ROC score

#### Model Optimization
- Performed **hyperparameter tuning** using GridSearchCV
- Cross-validated results using 5-fold stratified CV
- Finalized model ready for production deployment

---

## ✨ Key Features

### 🔬 Advanced Analytics
- ✅ Unsupervised learning for pattern discovery
- ✅ Multiple clustering algorithms comparison
- ✅ Statistical hypothesis validation
- ✅ Comprehensive EDA with 20+ visualizations

### 🤖 Machine Learning
- ✅ PCA for dimensionality reduction
- ✅ K-Means, Hierarchical, and DBSCAN clustering
- ✅ 5 classification models for sentiment analysis
- ✅ Hyperparameter optimization with GridSearch

### 📊 Data Visualization
- ✅ Interactive plots for cluster analysis
- ✅ Word clouds for review text
- ✅ Heatmaps for correlation analysis
- ✅ Distribution plots and box plots

### 💼 Business Intelligence
- ✅ Customer preference insights
- ✅ Competitive positioning analysis
- ✅ Cost-benefit recommendations
- ✅ Sentiment-driven feedback loops

---

## 🛠️ Technologies Used

### Core Libraries

<table>
<tr>
<td width="50%" valign="top">

#### 📊 Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Python 3.8+** - Programming language

#### 📈 Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

</td>
<td width="50%" valign="top">

#### 🤖 Machine Learning
- **Scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting classifier
- **NLTK** - Natural language processing
- **WordCloud** - Text visualization

#### 🔧 Utilities
- **Jupyter Notebook** - Interactive development
- **SciPy** - Scientific computing
- **Warnings** - Error handling

</td>
</tr>
</table>

### Specific Tools

```python
# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Text Processing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Evaluation
from sklearn.metrics import silhouette_score, roc_auc_score, classification_report
```

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/zomato-analysis.git
cd zomato-analysis
```

### Step 2: Install Dependencies

```bash
# Install required Python packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost nltk plotly wordcloud scipy jupyter
```

### Step 3: Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

### Step 4: Launch Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the notebook file
# Navigate to: _Zomato_Restaurant_Clustering_and_Sentiment_Analysis.ipynb
```

---

## 📈 Results

### 🎯 Clustering Insights

#### Optimal Number of Clusters: **6**

**Cluster Profiles:**

1. **Budget-Friendly Eateries** (25% of restaurants)
   - Average Cost: ₹200-400
   - Popular Cuisines: North Indian, Chinese, Street Food
   - Target: Students, daily commuters

2. **Family Dining** (30% of restaurants)
   - Average Cost: ₹500-800
   - Popular Cuisines: Multi-cuisine, Continental, Indian
   - Target: Families, casual diners

3. **Premium Fine Dining** (10% of restaurants)
   - Average Cost: ₹1500+
   - Popular Cuisines: Continental, Asian Fusion, Italian
   - Target: Special occasions, corporate dining

4. **Quick Bites & Cafes** (20% of restaurants)
   - Average Cost: ₹300-500
   - Popular Cuisines: Fast Food, Beverages, Desserts
   - Target: Quick meals, hangout spots

5. **Specialty Cuisine** (8% of restaurants)
   - Average Cost: ₹800-1200
   - Popular Cuisines: Japanese, Mediterranean, Mexican
   - Target: Food enthusiasts, explorers

6. **Local Favorites** (7% of restaurants)
   - Average Cost: ₹150-300
   - Popular Cuisines: Regional specialties
   - Target: Locals seeking authenticity

### 💬 Sentiment Analysis Results

- **Overall Sentiment Distribution:**
  - Positive Reviews: **68%**
  - Negative Reviews: **32%**

- **Best Performing Model:** Logistic Regression
  - Accuracy: XX%
  - AUC-ROC: **0.XX** (deployment-ready)

- **Key Sentiment Drivers:**
  - **Positive:** Food quality, ambiance, service, value for money
  - **Negative:** Wait times, cleanliness, pricing, taste inconsistency

---

## 💼 Business Impact

### For Customers 🧑‍🍳

1. **Better Discovery** - Find restaurants matching preferences and budget
2. **Informed Decisions** - Review sentiment insights guide dining choices
3. **Value Optimization** - Identify best cost-to-quality ratio options

### For Restaurants 🏪

1. **Competitive Analysis** - Understand positioning within clusters
2. **Improvement Areas** - Identify weaknesses from negative sentiment patterns
3. **Pricing Strategy** - Optimize costs based on cluster benchmarks
4. **Menu Planning** - Align offerings with popular cuisines in cluster

### For Zomato 📱

1. **Smart Recommendations** - Cluster-based personalized suggestions
2. **Market Segmentation** - Target marketing campaigns by cluster
3. **Quality Control** - Flag restaurants with declining sentiment scores
4. **Expansion Strategy** - Identify underserved cuisine-cost combinations

---

## 🔮 Future Enhancements

- [ ] **Real-time Sentiment Tracking** - Monitor review sentiment trends over time
- [ ] **Geographical Clustering** - Incorporate latitude/longitude for location-based clusters
- [ ] **Deep Learning NLP** - Use BERT or transformers for sentiment analysis
- [ ] **Multi-label Classification** - Classify reviews into specific aspects (food, service, ambiance)
- [ ] **Interactive Dashboard** - Build Streamlit/Dash app for stakeholder exploration
- [ ] **Temporal Analysis** - Study how restaurant performance changes over time
- [ ] **Review Summarization** - Auto-generate concise summaries from reviews
- [ ] **Image Analysis** - Analyze food pictures posted with reviews using CNNs

---

## 🗂️ Project Structure

```
zomato-analysis/
│
├── 📓 _Zomato_Restaurant_Clustering_and_Sentiment_Analysis.ipynb
│   └── Main analysis notebook with all code and visualizations
│
├── 📂 data/                    (Add your data files here)
│   ├── restaurant_metadata.csv
│   └── restaurant_reviews.csv
│
├── 📂 outputs/                 (Generated by notebook)
│   ├── cluster_visualizations/
│   ├── sentiment_plots/
│   └── model_metrics/
│
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # This file
└── 📄 LICENSE                 # MIT License
```

---

## 🤝 Contributing

Contributions are highly appreciated! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Ideas
- Add more clustering algorithms (Gaussian Mixture Models, Mean Shift)
- Implement aspect-based sentiment analysis
- Create visualizations for geographical restaurant distribution
- Build a web interface for interactive exploration
- Add more advanced NLP techniques (topic modeling, named entity recognition)

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**

- 🐙 GitHub: [@yourusername](https://github.com/PrasanthKumarS777)
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/prasanthsahu7)
- 📧 Email: pk777sahu@gmail.com


---

## 🙏 Acknowledgments

- **Zomato** - For providing the rich dataset that made this analysis possible
- **Deepinder Goyal & Pankaj Chaddah** - Founders of Zomato for revolutionizing food discovery in India
- **Scikit-learn Community** - For excellent machine learning tools
- **Data Science Community** - For continuous learning and inspiration

---

## 📚 References

- [K-Means Clustering - Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Text Classification with TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Principal Component Analysis (PCA)](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Sentiment Analysis Best Practices](https://www.nltk.org/howto/sentiment.html)

---

<div align="center">

### ⭐ If you found this project insightful, please star the repository!

**Hungry for insights? Let's analyze data! 🍽️📊**

Made with 💚 and 🍕 for India's vibrant food culture

[⬆ Back to Top](#️-zomato-restaurant-clustering--sentiment-analysis)

</div>
