# ==============================
# Amazon Bestselling Books ML Project
# ==============================

# --------- Import Libraries ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --------- Load Dataset ----------
df = pd.read_csv("bestsellers.with.categories.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nInfo:")
print(df.info())

# --------- Data Cleaning ----------
# Rename columns (easier to use)
df.columns = [
    "Name",
    "Author",
    "User_Rating",
    "Reviews",
    "Price",
    "Year",
    "Genre"
]

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# --------- Basic Statistics ----------
print("\nStatistical Summary:")
print(df.describe())

# --------- EDA Visualizations ----------
sns.set_style("darkgrid")

# Genre Count
plt.figure(figsize=(6,4))
sns.countplot(x="Genre", data=df)
plt.title("Fiction vs Non-Fiction Books")
plt.savefig("genre_distribution.png")
plt.show()

# Rating Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["User_Rating"], bins=20, kde=True)
plt.title("User Rating Distribution")
plt.savefig("rating_distribution.png")
plt.show()

# Price vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x="Price", y="User_Rating", hue="Genre", data=df)
plt.title("Price vs User Rating")
plt.savefig("price_vs_rating.png")
plt.show()

# Reviews vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x="Reviews", y="User_Rating", hue="Genre", data=df)
plt.title("Reviews vs User Rating")
plt.savefig("reviews_vs_rating.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,5))
corr = df[["User_Rating","Reviews","Price","Year"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# --------- Machine Learning Model ----------
print("\n===== MACHINE LEARNING: GENRE CLASSIFIER =====")

# Encode Genre (Target variable)
le = LabelEncoder()
df["Genre_encoded"] = le.fit_transform(df["Genre"])
# Fiction = 0, Non-Fiction = 1 (may vary)

# Features & Target
X = df[["User_Rating", "Reviews", "Price", "Year"]]
y = df["Genre_encoded"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# --------- Evaluation ----------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# --------- Save Model ----------
joblib.dump(model, "genre_classifier_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as genre_classifier_model.pkl")
print("Scaler saved as scaler.pkl")

# --------- Example Prediction ----------
print("\nExample Prediction:")

sample_book = [[4.8, 15000, 10, 2018]]  # rating, reviews, price, year
sample_scaled = scaler.transform(sample_book)

prediction = model.predict(sample_scaled)

genre_result = le.inverse_transform(prediction)

print("Predicted Genre:", genre_result[0])
