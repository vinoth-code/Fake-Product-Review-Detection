# Fake-Product-Review-Detection
# 🧠 Fake Product Review Detection using Machine Learning

This project is a Streamlit web app that detects whether a product review is **genuine** or **fake**, using a Logistic Regression model trained on real labeled review data.

---

## 📌 Objective

To build a machine learning-based application that identifies fake product reviews based on their text content. The system helps improve trust and decision-making for e-commerce platforms.

---

## 🚀 Features

- Predicts if a review is **fake** or **genuine**
- Displays **model confidence score**
- **Live deployment** on Streamlit Cloud
- Model hosted on Google Drive and loaded with `gdown`

---

## 🛠️ Tools & Technologies Used

| Tool          | Purpose                          |
|---------------|----------------------------------|
| Python        | Main language                    |
| Pandas        | Data processing                  |
| scikit-learn  | Machine learning model & tools   |
| NLTK          | Text cleaning and lemmatization  |
| TF-IDF        | Text to feature vector conversion|
| Streamlit     | Web app interface                |
| Google Drive  | Remote hosting of `.pkl` files   |
| gdown         | Download files in app runtime    |

---

## 🧠 ML Workflow

1. **Dataset Cleaning:** Stopword removal, lowercasing, punctuation removal, lemmatization
2. **Feature Engineering:** TF-IDF Vectorizer (`max_features=1000`)
3. **Model:** Logistic Regression classifier
4. **Output:** Saved model and vectorizer (`model.pkl`, `vectorizer.pkl`)
5. **Web App:** Built using Streamlit and deployed via Streamlit Cloud

---

## 📁 Folder Structure

fake-review-detector/
├── app.py # Streamlit UI
├── requirements.txt # Python dependencies
└── (model.pkl, vectorizer.pkl hosted on Google Drive)


---

## 💡 Sample Predictions

| Input Review                                              | Output     |
|-----------------------------------------------------------|------------|
| "The product is excellent and delivery was quick."        | ✅ Genuine |
| "Good item camera battery price quality performance"      | ❌ Fake    |

---

## ⚙️ How to Run Locally

1. Clone the repository  
2. Install dependencies  
```bash
pip install -r requirements.txt
-----
3. RUN prepare_app.py
4)run app.py

streamlit run app.py

