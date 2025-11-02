
# 🌍 Tourism Recommender System – India

An intelligent **Streamlit web app** that recommends Indian tourist destinations, predicts accessibility, shows real-time weather, and provides travel booking links — all in one place!

---

## 🚀 Features

* 🧭 **Accessibility Checker** – Predicts how accessible a destination is using a trained Random Forest Classifier.
* 📊 **Region + State Popular Destinations** – Displays the most popular places in each region/state based on attractions.
* 🎯 **Content-Based Destination Recommender** – Suggests similar destinations using TF-IDF and cosine similarity.
* 🗂 **Multi-Class Category Classifier** – Predicts the type/category of a destination (e.g., Hill Station, Beach, Heritage, etc.).
* ⛅ **3-Day Real-Time Weather Forecast** – Shows live weather data for any Indian city/destination.
* 📸 **Destination Image Explorer** – Fetches real images using the Unsplash API.
* ✈️🚆 **Travel Booking Links** – Quick links for flights (Skyscanner) and trains (IRCTC).

---

## 🧠 Machine Learning Models Used

* **Random Forest Classifier** – for predicting Accessibility and Category.
* **TF-IDF Vectorizer + Cosine Similarity** – for content-based recommendations.

---

## 📂 Project Structure

```
Tourism-Recommender-System/
│
├── data/
│   ├── Expanded_Indian_Travel_Dataset.csv
│   ├── holidify.csv
│
├── utils/
│   ├── weather_api.py
│
├── myapp.py                # Streamlit main app
├── requirements.txt        # Dependencies
└── README.md               # Project overview
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/lakhbir-05/TOURISM-RECOMMENDATION-SYSTEM---INDIA.git
   cd TOURISM-RECOMMENDATION-SYSTEM---INDIA
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run myapp.py
   ```

---

## 🔑 API Keys Required

* **Unsplash API Key** → for fetching destination images
* **OpenWeatherMap API Key** → used inside `utils/weather_api.py` for real-time weather

---

**APP LINK** :https://tourism-recommendation-system---india-rnjyanf9d8zuhnorzahkmm.streamlit.app/


