import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
csv_path = os.path.join(path, "dataset.csv")

df = pd.read_csv(csv_path, encoding="latin-1", engine="python", on_bad_lines="skip")
print("First 5 records:", df.head())
print("Data types:", df.dtypes)
print("Missing values:", df.isnull().sum()) 
print("Info:", df.info())



# -------------- Phase 1 ------------
from sklearn.preprocessing import MinMaxScaler

audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

X = df[audio_features].copy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Shape:", X_scaled.shape)  # (113999, 9)


# -------------- Phase 2 ------------

from sklearn.neighbors import NearestNeighbors

# Fit a KNN model using cosine distance
knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
knn.fit(X_scaled)

print("KNN model ready.")

# -------------- Phase 3 ------------

def recommend(track_name, df, X_scaled, knn, n=10):
    # Find the index of the song
    matches = df[df['track_name'].str.lower() == track_name.lower()]
    
    if matches.empty:
        print(f"Track '{track_name}' not found.")
        return
    
    idx = matches.index[0]
    
    # Get the feature vector for this song
    song_vector = X_scaled[idx].reshape(1, -1)
    
    # Find nearest neighbors
    distances, indices = knn.kneighbors(song_vector)
    
    # Skip index 0 — that's the song itself
    rec_indices = indices[0][1:n+1]
    rec_distances = distances[0][1:n+1]
    
    results = df.iloc[rec_indices][['track_name', 'artists', 'track_genre']].copy()
    results['similarity'] = (1 - rec_distances).round(3)
    
    print(f"\nRecommendations for: '{df.iloc[idx]['track_name']}' by {df.iloc[idx]['artists']}")
    print(f"Genre: {df.iloc[idx]['track_genre']}\n")
    print(results.to_string(index=False))

# Try it out
recommend("Blinding Lights", df, X_scaled, knn)

# ------ Phase 4 ------------

def profile_comparison(track_name, df, X_scaled, knn, n=5):
    matches = df[df['track_name'].str.lower() == track_name.lower()]
    if matches.empty:
        return
    
    idx = matches.index[0]
    song_vector = X_scaled[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(song_vector)
    
    rec_indices = indices[0][1:n+1]
    
    cols = ['track_name'] + audio_features
    query_row = df.iloc[[idx]][cols]
    rec_rows = df.iloc[rec_indices][cols]
    
    comparison = pd.concat([query_row, rec_rows])
    comparison.index = ['[Query]'] + [f'Rec {i+1}' for i in range(n)]
    print(comparison.round(3).to_string())

profile_comparison("Blinding Lights", df, X_scaled, knn)