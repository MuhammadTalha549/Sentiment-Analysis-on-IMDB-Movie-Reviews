# Sentiment-Analysis-on-IMDB-Movie-Reviews

# 🎬 IMDB Movie Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-purple.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📋 Project Overview

This project focuses on **Sentiment Analysis** of IMDB movie reviews using Natural Language Processing (NLP) techniques. It classifies reviews as **positive** or **negative**, applying traditional machine learning models and custom preprocessing pipelines to handle raw text data. The dataset is split into training, testing, and validation sets for a comprehensive evaluation.

## 🎯 Objectives

* Process and clean raw IMDB review text
* Convert text into numerical features using TF-IDF and Count Vectorizer
* Train models such as **Logistic Regression**, **Naive Bayes**, or **SVM**
* Evaluate performance with metrics like Accuracy, Precision, Recall, and F1-Score
* Allow **CLI-based sentiment prediction** for new reviews

## 📊 Dataset

**IMDB Large Movie Review Dataset**

* **Source**: \[Kaggle / Stanford AI Lab]
* **Total Reviews**: 50,000
* **Split**: 25,000 for training, 25,000 for testing (balanced dataset)
* **Format**: Pre-separated into `train.csv`, `test.csv`, and optionally `valid.csv`
* **Columns**:

  * `review`: Raw text of the review
  * `sentiment`: Target label (`positive` or `negative`)

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or Python CLI
```

### Required Libraries

```bash
pip install numpy pandas scikit-learn nltk matplotlib seaborn
```

### Clone the Repository

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

## 📁 Project Structure

```
imdb-sentiment-analysis/
├── README.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── valid.csv
├── models/
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
├── scripts/
│   ├── preprocess.py              # Text cleaning and tokenization
│   ├── train_model.py             # Model training and saving
│   └── test_review_cli.py         # CLI-based sentiment prediction
├── notebooks/
│   └── analysis.ipynb             # Full EDA and training notebook
└── requirements.txt
```

## 🚀 How to Run

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Open `notebooks/analysis.ipynb` and run the cells step-by-step for preprocessing, training, and evaluation.

### Option 2: Train Model via Script

```bash
python scripts/train_model.py
```

### Option 3: Predict Sentiment from CLI

```bash
python scripts/test_review_cli.py
```

Sample input:

```
Enter a review: This movie was absolutely wonderful!
Predicted Sentiment: Positive
```

## 🧹 Text Preprocessing Pipeline

* Lowercasing
* HTML tag removal
* Punctuation & digit removal
* Stopword filtering (using NLTK)
* Tokenization
* TF-IDF or Count Vectorization

## 🤖 Models Used

* **Logistic Regression**
* **Multinomial Naive Bayes**
* (Optional) SVM, Random Forest, or LSTM (if implemented)

## 📈 Evaluation Metrics

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 89.2%    | 0.89      | 0.89   | 0.89     |
| Naive Bayes         | 86.4%    | 0.87      | 0.86   | 0.86     |

*Results based on test dataset*

## 📊 Visualizations

* **Word Cloud** of frequent terms in positive/negative reviews
* **Confusion Matrix** for each model
* **Bar plots** of precision, recall, F1-score
* **ROC Curve** and **AUC** (if binary classifier)

## 🔍 Technical Highlights

* Clean and modular code using `.py` scripts and `.ipynb` notebooks
* Model persistence using `joblib` for later use in CLI
* Custom text preprocessing pipeline using `nltk` and `re`
* Vectorization using both `CountVectorizer` and `TfidfVectorizer`

## 🚧 Limitations & Future Work

### Limitations

* Limited to binary sentiment classification
* No hyperparameter optimization or cross-validation (yet)

### Future Enhancements

* Integrate **deep learning** models like LSTM or BERT
* Add **streamlit or flask web app** for UI
* Perform **hyperparameter tuning** (GridSearchCV, RandomizedSearchCV)
* Add **cross-validation** and ensemble methods

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for:

* Additional model implementations
* Improved preprocessing techniques
* Enhanced CLI or frontend integration


## 📧 Contact

**Your Name**

* GitHub: [@https://github.com/MuhammadTalha549](https://github.com/MuhammadTalha549)
* Email: [@talhamuhammad549@gmail.com](talhamuhammad549@gmail.com)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
