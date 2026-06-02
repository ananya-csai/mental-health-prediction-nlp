# 🧠 Mental Health Prediction using NLP and Machine Learning

## 📌 Project Overview

Mental health disorders such as depression have become increasingly prevalent worldwide. Early detection and intervention can significantly improve outcomes and provide timely support to individuals at risk.

This project leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically identify depression-related patterns from textual social media content. Using a dataset of Reddit posts, multiple machine learning models were trained and evaluated to classify whether a given text indicates signs of depression.

The project includes:

- Exploratory Data Analysis (EDA)
- Text Preprocessing and Cleaning
- TF-IDF Feature Engineering
- Machine Learning Model Training
- Model Evaluation and Comparison
- Streamlit-based Prediction Application
- Model Serialization using Pickle

---

## 🎯 Objectives

- Analyze depression-related textual data from Reddit.
- Explore patterns and characteristics within mental health discussions.
- Convert unstructured text into numerical representations using TF-IDF.
- Train and compare multiple machine learning algorithms.
- Build an interactive application for real-time depression prediction.
- Demonstrate practical applications of NLP in healthcare and mental health analytics.

---

## 📂 Dataset Information

**Dataset:** Reddit Depression Dataset

| Feature | Description |
|----------|------------|
| clean_text | Preprocessed Reddit post/comment text |
| is_depression | Target variable (0 = Non-Depression, 1 = Depression) |

### Dataset Statistics

- Total Records: **7,731**
- Features: **2**
- Classes: Balanced Dataset
    - Non-Depression: 3,900
    - Depression: 3,831

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Natural Language Processing

- Regular Expressions (Regex)
- TF-IDF Vectorization

### Machine Learning

- Multinomial Naive Bayes
- Logistic Regression
- Random Forest Classifier

### Model Deployment

- Streamlit
- Pickle

### Development Environment

- Jupyter Notebook

---

## 🔄 Project Workflow

### 1️⃣ Data Collection

- Loaded Reddit depression dataset.
- Verified dataset dimensions and structure.

### 2️⃣ Exploratory Data Analysis (EDA)

Performed:

- Dataset overview
- Class distribution analysis
- Text length analysis
- Depression vs Non-Depression comparison

### 3️⃣ Text Preprocessing

Implemented:

- Lowercase conversion
- URL removal
- Special character removal
- Text cleaning using Regular Expressions

### 4️⃣ Feature Engineering

Used:

```python
TF-IDF Vectorizer
```

Configuration:

```python
max_features = 5000
```

Resulting Feature Matrix:

```text
(7731, 5000)
```

### 5️⃣ Model Training

Three machine learning models were trained:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Random Forest Classifier

### 6️⃣ Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 7️⃣ Deployment

- Trained model serialized using Pickle.
- Interactive Streamlit application developed for real-time predictions.

---

## 📊 Exploratory Data Analysis

### Class Distribution

The dataset is well balanced between depression and non-depression classes.

### Text Length Analysis

Analyzed post lengths to understand textual behavior across both classes.

### Depression vs Text Length

Observed noticeable differences in text length distributions between depression-related and non-depression posts.

---

## 🤖 Model Performance

| Model | Accuracy |
|---------|----------|
| Multinomial Naive Bayes | 88.88% |
| Logistic Regression | 95.47% |
| Random Forest Classifier | 95.47% |

### Best Performing Models

🏆 Logistic Regression

🏆 Random Forest Classifier

Both achieved approximately:

```text
95.47% Accuracy
```

---

## 📈 Confusion Matrix

The confusion matrix was generated to analyze prediction performance and identify false positives and false negatives.

The model demonstrated strong classification capability with high precision and recall across both classes.

---

## 🚀 Streamlit Application

An interactive web application was built using Streamlit.

### Features

- User enters text input.
- Model preprocesses and vectorizes text.
- Prediction generated instantly.
- Displays whether the text indicates depression-related patterns.

### Run Locally

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
mental-health-prediction-nlp
│
├── data
│   └── depression_dataset_reddit_cleaned.csv
│
├── notebooks
│   └── Mental_Health_Prediction.ipynb
│
├── models
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app
│   └── app.py
│
├── reports
│   ├── final_report.pdf
│   └── presentation.pptx
│
├── images
│   ├── class_distribution.png
│   ├── text_length_distribution.png
│   └── confusion_matrix.png
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 🔍 Key Findings

- Depression-related text can be effectively classified using NLP techniques.
- TF-IDF provides meaningful numerical representations of textual data.
- Logistic Regression and Random Forest delivered excellent performance.
- Machine Learning can assist in early identification of mental health concerns.
- NLP has significant potential in healthcare analytics and mental health monitoring.

---

## 💡 Future Improvements

- Incorporate advanced NLP models such as:
  - BERT
  - RoBERTa
  - DistilBERT

- Deploy application on:
  - Streamlit Cloud
  - Render
  - Hugging Face Spaces

- Add sentiment analysis capabilities.

- Integrate explainable AI techniques for model interpretability.

- Build a full-scale mental health assistance platform.

---

## 📚 Skills Demonstrated

- Data Analysis
- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Text Classification
- Machine Learning
- Model Evaluation
- Feature Engineering
- Streamlit Development
- Data Visualization
- Python Programming

---

## 👩‍💻 Author

**Ananya Shukla**

Computer Science Undergraduate | AI/ML & NLP Enthusiast

Interested in:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Research-Oriented Projects

---

⭐ If you found this project interesting, feel free to star the repository.
