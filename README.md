# generate_readme.py

readme_content = r"""
# 📚 Amazon Bestselling Books — Data Analysis & Genre Prediction (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview
This project performs Exploratory Data Analysis (EDA) and builds a Machine Learning classification model on Amazon’s Top 50 Bestselling Books dataset (2009–2019).

The objective is to analyze reader behavior and predict whether a book will belong to the Fiction or Non-Fiction category using measurable attributes such as ratings, reviews, price, and year.

Workflow:
raw dataset → data cleaning → visualization → model training → evaluation

---

## 🎯 Problem Statement
Can we predict the genre of a bestselling book using numerical features?

Features:
- User Rating
- Review Count
- Price
- Bestseller Year

Target:
- Genre (Fiction / Non-Fiction)

---

## 📊 Dataset Information

| Attribute | Description |
|----------|------------|
| Name | Book title |
| Author | Author name |
| User_Rating | Average rating (0–5) |
| Reviews | Total number of reviews |
| Price | Book price |
| Year | Bestseller year |
| Genre | Fiction or Non-Fiction |

Records: 550+ books

---

## 🛠️ Technology Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 📈 Visual Results

### Genre Distribution
![Genre](images/genre_distribution.png)

### Price vs Rating
![PriceRating](images/price_vs_rating.png)

### Reviews vs Rating
![ReviewsRating](images/reviews_vs_rating.png)

### Correlation Heatmap
![Heatmap](images/correlation_heatmap.png)

### Confusion Matrix
![CM](images/confusion_matrix.png)

---

## 🤖 Machine Learning Model

Algorithm Used: Logistic Regression

Preprocessing:
- Removed duplicates
- Encoded labels
- Feature scaling
- 80/20 train-test split

Input Features:
- User_Rating
- Reviews
- Price
- Year

Target:
0 → Fiction
1 → Non-Fiction

---

## ▶️ Running the Project

Clone:
git clone https://github.com/PalakRay07/Amazon-Books-ML.git

Install:
pip install pandas numpy matplotlib seaborn scikit-learn joblib

Run:
python book_analysis_ml.py

---

## 📂 Project Structure
Amazon-Books-ML/
│
├── bestsellers.with.categories.csv
├── book_analysis_ml.py
├── genre_classifier_model.pkl
├── scaler.pkl
├── images/
└── README.md

---

## 💡 Future Improvements
- Recommendation system
- Streamlit web app
- Cloud deployment
- Advanced ML models

---

## 👩‍💻 Author
Palak.  
AI/ML Intern  
palak070704@gmail.com
"""

# Write to README.md
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README.md file generated successfully!")
