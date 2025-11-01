
import joblib
import pandas as pd

# Load models
clf = joblib.load("models/accessibilty_model.pkl")
encoders = joblib.load("models/feature_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
cosine_sim = joblib.load("models/cosine_similarity.pkl")


# Load datasets
df = pd.read_csv("data/Expanded_Indian_Travel_Dataset.csv")
df_most = pd.read_csv("data/holidify.csv")

# Accessibility Prediction
def predict_accessibility(Destination,state, region, category, airport, railway):
    features = pd.DataFrame([[state, region, category, airport, railway]],
                            columns=["Destination","State","Region","Category","Nearest Airport","Nearest Railway Station"])
    for col in features.columns:
        features[col] = encoders[col].transform(features[col])
    pred = clf.predict(features)
    return target_encoder.inverse_transform(pred)[0]

# Region + State Popularity Recommender
def recommend_by_region_state(region, state, top_n=5):
    region_df = df[(df["Region"].str.lower()==region.lower()) & (df["State"].str.lower()==state.lower())]
    region_df["PopularityScore"] = region_df["Popular Attraction"].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
    )
    top_dest = region_df.sort_values(by="PopularityScore", ascending=False)
    return top_dest[["Destination Name","Category","Popular Attraction","PopularityScore"]].head(top_n)

# Content-Based Recommendation
indices = pd.Series(df.index, index=df["Destination Name"].str.lower())
def recommend_similar_destinations(name, top_n=5):
    name = name.lower()
    if name not in indices:
        return ["Destination not found."]
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)[1:top_n+1]
    dest_indices = [i[0] for i in sim_scores]
    return df["Destination Name"].iloc[dest_indices].tolist()
