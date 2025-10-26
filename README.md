# 📧 Spam Detector AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A simple yet powerful web application built with **Streamlit** and **Scikit-learn** to classify SMS/email messages as **Spam** or **Ham** (Not Spam).

[Live Demo](https://spam-detector-app-5uswehehzoruhnftj29ckv.streamlit.app/) • [Report Bug](https://github.com/Nikhil-Singh2745/spam-detector-streamlit/issues) • [Request Feature](https://github.com/Nikhil-Singh2745/spam-detector-streamlit/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Technology Stack](#-technology-stack)
- [Model Performance](#-model-performance)
- [Setup and Usage](#-setup-and-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project demonstrates an **end-to-end Machine Learning workflow** for text classification. The application analyzes text messages and predicts whether they are spam or legitimate messages using advanced NLP techniques.

### Workflow:

1. 📊 **Data Loading** - Loading and inspecting SMS message data
2. 🔍 **EDA** - Exploratory Data Analysis including class distribution and message length analysis
3. 🔤 **Feature Extraction** - Using TF-IDF (Term Frequency-Inverse Document Frequency) with built-in preprocessing
4. 🤖 **Model Training** - Training and comparing Logistic Regression and Multinomial Naive Bayes models
5. 📈 **Evaluation** - Assessing models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices
6. 🚀 **Deployment** - Deploying the best-performing model as an interactive web app using Streamlit

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| ⚡ **Real-time Prediction** | Classifies input text as Spam or Ham instantly |
| 🎯 **Confidence Score** | Displays the model's prediction probability |
| 📊 **Input Analysis** | Shows message characteristics (length, word count, % uppercase, % digits) |
| 🔍 **Keyword Highlighting** | Identifies potential common spam keywords in the input |
| 📝 **Example Messages** | Quick-load buttons for sample Spam and Ham messages |
| 📈 **Performance Metrics** | Displays key model performance scores in the sidebar |

---

## 📸 Screenshots

### Ham (Not Spam) Prediction
![Ham Prediction Example](https://github.com/Nikhil-Singh2745/spam-detector-streamlit/blob/main/screenshots/screenshot3.png)
*Example of a legitimate message being correctly classified*

### Spam Prediction with Analysis
![Spam Prediction Example](https://github.com/Nikhil-Singh2745/spam-detector-streamlit/blob/main/screenshots/screenshot2.png)
*Spam message detection with detailed analysis and keyword highlighting*


## 🛠 Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) |
| **Machine Learning** | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) |
| **Web Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white) |

</div>

### Key Libraries:
- **Scikit-learn**: `TfidfVectorizer`, `LogisticRegression`, `MultinomialNB`, evaluation metrics
- **Streamlit**: Interactive web interface
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization (in notebook)
- **WordCloud**: Visual representation of text data

---

## 📊 Model Performance

Performance metrics on the **Test Set** using **Logistic Regression**:

```
📈 Model Performance Metrics
├── Accuracy:        96.7%
├── Spam F1-Score:   0.87
└── Ham F1-Score:    0.98
```

<div align="center">

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.7% |
| **Precision (Spam)** | ~0.87 |
| **Recall (Spam)** | ~0.87 |
| **F1-Score (Spam)** | 0.87 |
| **F1-Score (Ham)** | 0.98 |

</div>

> **Note**: These values are based on the training run without advanced NLTK preprocessing. The model shows excellent performance in identifying legitimate messages (Ham) and good performance in detecting spam.

---

## 🚀 Setup and Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nikhil-Singh2745/spam-detector-streamlit.git
   cd spam-detector-streamlit
   ```

2. **Create and activate a virtual environment**:
   
   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to:
   ```
   http://localhost:8501
   ```

---

## 📁 Project Structure

```
spam-detector-streamlit/
│
├── app.py                      # Main Streamlit application
├── model.pkl                   # Trained Logistic Regression model
├── vectorizer.pkl              # Trained TF-IDF vectorizer
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── notebooks/                  # Jupyter notebooks (optional)
│   └── training.ipynb         # Model training and EDA
│
├── data/                       # Dataset folder (optional)
│   └── spam.csv               # SMS spam dataset
│
└── screenshots/                # Application screenshots
    ├── screenshot1.png
    ├── screenshot2.png
    └── screenshot3.png
```

---

## 🔧 How It Works

### 1. Text Preprocessing
```python
# Built-in TF-IDF preprocessing:
- Converts text to lowercase
- Removes stop words
- Tokenizes text
- Calculates TF-IDF scores
```

### 2. Feature Extraction
The TF-IDF vectorizer transforms text into numerical features that the model can understand.

### 3. Classification
The trained Logistic Regression model predicts whether the message is spam or ham based on the extracted features.

### 4. Analysis
The app provides additional insights:
- Message length
- Word count
- Percentage of uppercase letters
- Percentage of digits
- Spam keyword detection

---

</div>
