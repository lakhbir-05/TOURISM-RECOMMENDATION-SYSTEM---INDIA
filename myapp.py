import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from utils.weather_api import get_forecast

# --- Additional imports for Map & Image Explorer ---
from geopy.geocoders import Nominatim
import requests

# ==========================
# Load Data
# ==========================
df1 = pd.read_csv("data/Expanded_Indian_Travel_Dataset.csv")
df2 = pd.read_csv("data/holidify.csv")

# ==========================
# Accessibility Classifier
# ==========================
def train_accessibility_classifier(df1):
    df = df1.copy()
    df = df.dropna(subset=["Accessibility"])
    features = ["State", "Region", "Nearest Airport", "Nearest Railway Station"]
    X = df[features].fillna("Unknown")

    encoders = {}
    for col in features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = df["Accessibility"]
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, encoders, le_y

access_model, encoders, le_access = train_accessibility_classifier(df1)

# ==========================
# Booking Links Helper
# ==========================
def get_flight_booking_link(airport_name):
    if pd.isna(airport_name) or airport_name.strip() == "":
        return None
    query = airport_name.replace(" ", "+")
    return f"https://www.skyscanner.co.in/transport/flights-to/{query}"

def get_train_booking_link(station_name):
    if pd.isna(station_name) or station_name.strip() == "":
        return None
    return "https://www.irctc.co.in/nget/train-search"

# ==========================
# Content-Based Recommender
# ==========================
def content_based_recommender(destination, top_n=5):
    df = df1.copy()
    df["combined"] = (
        df["Destination Name"].fillna("") + " " +
        df["Category"].fillna("") + " " +
        df["Popular Attraction"].fillna("")
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined"])

    idx = df[df["Destination Name"].str.lower() == destination.lower()].index
    if len(idx) == 0:
        return pd.DataFrame()

    idx = idx[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    df["similarity"] = cosine_sim
    return df.sort_values("similarity", ascending=False).head(top_n)[
        ["Destination Name", "State", "Category", "Popular Attraction", "Nearest Airport", "Nearest Railway Station"]
    ]

# ==========================
# Multi-Class Category Classifier
# ==========================
def train_category_classifier():
    df = df1.dropna(subset=["Category"])
    features = ["State", "Region", "Accessibility"]
    X = df[features].fillna("Unknown")

    encoders = {}
    for col in features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    y = df["Category"]
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, encoders, le_y

category_model, cat_encoders, le_cat = train_category_classifier()

# ==========================
#  Image Explorer Setup
# ==========================
UNSPLASH_ACCESS_KEY = "Pc_kgUGh2MIOp_JMwde-p1EavBrwPcHBYYbcODQCB64"
geolocator = Nominatim(user_agent="tourism_app")

def get_lat_lon(destination, state=None):
    try:
        query = f"{destination}, {state}, India" if state else f"{destination}, India"
        location = geolocator.geocode(query)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

def get_destination_image(destination):
    url = f"https://api.unsplash.com/search/photos?query={destination}&client_id={UNSPLASH_ACCESS_KEY}&per_page=1"
    response = requests.get(url).json()
    if response.get("results"):
        return response["results"][0]["urls"]["regular"]
    return None

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="Tourism Recommender", layout="wide")
st.title("🌍 Tourism Recommender System - India")


menu = [
    "Home",
    "Accessibility checker",
    "Region + State Popular Destinations",
    "Content-Based Destination ",
    "Category Classifier",
    " 3 days Real-Time Weather",
    "To explore images of Destination"
    
]



choice = st.sidebar.radio("📌 Select Feature", menu)

# ---------------- Home ----------------
if choice == "Home":
    st.subheader("Welcome! 🎉")
    st.markdown("""
    🚦Accessibility Checker + Nearest Airport/Railway    
    📊 Region + State Popular Destinations 
    🎯 Content-Based destination           
    📂 Multi-Class Category Classifier  
    ⛅ 3 days Real-Time Weather  
    🌅 To explore images of destination
      
    """)


# ---------------- Accessibility Classifier ----------------
elif choice == "Accessibility checker":
    st.subheader("🚦 Accessibility Predictor")

    state = st.selectbox("Select State", sorted(df1["State"].unique()))
    destinations = sorted(df1[df1["State"] == state]["Destination Name"].dropna().unique())
    destination = st.selectbox("Select Destination", destinations)

    if destination:
        dest_info = df1[(df1["State"] == state) & (df1["Destination Name"] == destination)].iloc[0]
        region = dest_info["Region"]
        airport = dest_info["Nearest Airport"]
        station = dest_info["Nearest Railway Station"]

        st.info(f"Region: {region}")
        st.info(f"Nearest Airport: {airport}")
        st.info(f"Nearest Railway Station: {station}")

        # --- Predict Accessibility first ---
        if st.button("Predict Accessibility"):
            x = pd.DataFrame([[state, region, airport, station]],
                             columns=["State", "Region", "Nearest Airport", "Nearest Railway Station"])
            for col in x.columns:
                x[col] = encoders[col].transform(x[col])
            pred = access_model.predict(x)[0]
            pred_label = le_access.inverse_transform([pred])[0]
            st.success(f"Predicted Accessibility: {pred_label}")

        # --- Show Travel Booking Links below the prediction ---
        st.markdown("### ✈️ Travel Booking Links")
        flight_link = get_flight_booking_link(airport)
        train_link = get_train_booking_link(station)

        if flight_link:
            st.markdown(f"[🛫 Book Flights via Skyscanner]({flight_link})", unsafe_allow_html=True)
        else:
            st.warning("No airport information available.")

        if train_link:
            st.markdown(f"[🚆 Book Trains on IRCTC]({train_link})", unsafe_allow_html=True)
        else:
            st.warning("No railway station information available.")


# ---------------- Region + State Popularity Recommender ----------------
elif choice == "Region + State Popular Destinations":
    st.subheader("📊 Region + State Popularity Recommender")
    state = st.selectbox("Select State", sorted(df1["State"].unique()))
    region = df1.loc[df1["State"] == state, "Region"].iloc[0] if not df1[df1["State"] == state].empty else "Unknown"
    st.info(f"Region: {region}")

    categories = sorted(df1[df1["State"] == state]["Category"].dropna().unique())
    selected_categories = st.multiselect("Filter by Category", categories, default=[])
    filtered_df = df1[df1["State"] == state]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    recs = filtered_df.sort_values(by="Popular Attraction", ascending=False).head(10)
    if not recs.empty:
        st.write("### 🏆 Top Destinations")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['Destination Name']}** ({row['Category']}) — {row['Popular Attraction']}")
    else:
        st.warning("No destinations found for this selection.")

# ---------------- Content-Based Destination Recommender ----------------
elif choice == "Content-Based Destination ":
    st.subheader("🎯 Content-Based Destination Recommender")
    all_destinations = sorted(df1["Destination Name"].dropna().unique())
    destination = st.selectbox("Select a Destination", all_destinations)
    if st.button("Get Similar Destinations"):
        recs = content_based_recommender(destination, top_n=5)
        if not recs.empty:
            st.write("### 🔍 Similar Destinations")
            for _, row in recs.iterrows():
                st.markdown(f"- **{row['Destination Name']}** ({row['Category']}) — {row['Popular Attraction']}")
        else:
            st.warning("No similar destinations found.")

# ---------------- Category Classifier ----------------
elif choice == "Category Classifier":
    st.subheader("📂 Multi-Class Category Classifier")
    state = st.selectbox("Select State", sorted(df1["State"].unique()))
    filtered_df = df1[df1["State"] == state]
    region = filtered_df["Region"].iloc[0] if not filtered_df.empty else "Unknown"
    accessibility = filtered_df["Accessibility"].iloc[0] if not filtered_df.empty else "Unknown"
    st.info(f"Region: {region}")
    st.info(f"Accessibility: {accessibility}")
    if st.button("Predict Category"):
        if filtered_df.empty:
            st.warning("No data available for this state.")
        else:
            x = pd.DataFrame([[state, region, accessibility]], columns=["State", "Region", "Accessibility"])
            for col in x.columns:
                x[col] = cat_encoders[col].transform(x[col])
            pred = category_model.predict(x)[0]
            category = le_cat.inverse_transform([pred])[0]
            st.success(f"Predicted Category: {category}")

# ---------------- Real-Time Weather ----------------
elif choice == " 3 days Real-Time Weather":
    st.subheader("⛅ Real-Time Weather Report")
    place = st.text_input("Enter City / Destination")
    if st.button("Get Weather"):
        if place:
            forecast = get_forecast(place)
            if "error" in forecast:
                st.error(forecast["error"])
            else:
                st.write(f"### 3-Day Weather Forecast for {place}")
                for entry in forecast[:9]:
                    st.info(f"{entry['datetime']} | 🌡 {entry['temp']}°C | {entry['weather']}")

                # Add booking links if destination found
                match = df1[df1["Destination Name"].str.lower() == place.lower()]
                if not match.empty:
                    airport = match.iloc[0]["Nearest Airport"]
                    station = match.iloc[0]["Nearest Railway Station"]

                    st.markdown("### 🚉 Quick Booking Links")
                    st.markdown(f"[🛫 Book Flights via Skyscanner]({get_flight_booking_link(airport)})", unsafe_allow_html=True)
                    st.markdown(f"[🚆 Book Trains on IRCTC]({get_train_booking_link(station)})", unsafe_allow_html=True)


# ---------------- Destination Image Explorer ----------------
elif choice == "To explore images of Destination":
    st.subheader("📸 Destination Image Explorer")

    place = st.text_input("Enter Destination Name (e.g., Jaipur, Manali, Goa)")
    num_images = st.slider("Number of Images", min_value=1, max_value=10, value=5)

    if st.button("Show Images"):
        if place.strip() == "":
            st.warning("Please enter a valid destination name.")
        else:
            url = f"https://api.unsplash.com/search/photos?query={place}&client_id={UNSPLASH_ACCESS_KEY}&per_page={num_images}"
            response = requests.get(url).json()

            if response.get("results"):
                st.markdown(f"### 🌅 Showing {len(response['results'])} images of **{place.title()}**")
                cols = st.columns(2)
                for idx, img in enumerate(response["results"]):
                    with cols[idx % 2]:
                        st.image(img["urls"]["regular"], caption=img["alt_description"] or place, use_container_width=True)
            else:
                st.error("No images found for this destination.")



