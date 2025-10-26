import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import fcluster

pt = joblib.load('models/power_transformer.pkl')
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca_model.pkl')
kmeans = joblib.load('models/kmeans_model.pkl')
dbscan = joblib.load('models/dbscan_model.pkl')
Z = joblib.load('models/hierarchical_linkage.pkl')    

df = pd.read_csv("database.csv")
df.dropna(axis=1, inplace=True)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M:%S')
df.dropna(subset=['Date', 'Time'], inplace=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

features = ['Latitude', 'Longitude', 'Depth', 'Magnitude',
            'Type', 'Source', 'Magnitude Source', 'Status',
            'Year', 'Month']

numeric_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
categorical_cols = ['Type', 'Source', 'Magnitude Source', 'Status']

X = df[features].copy()
for col in numeric_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X[col] = X[col].clip(lower, upper)

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

X[numeric_cols] = pt.transform(X[numeric_cols])
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

st.title("🌎 Earthquake Clustering Dashboard")
st.write("This app reproduces clustering results using consistent preprocessing and saved models.")

st.subheader("📋 First 20 Records (Processed Data)")
st.dataframe(df.head(20)[features])

st.subheader("🧾 Input New Earthquake Record")
input_data = {}

for col in numeric_cols:
    val = float(df[col].mean())
    input_data[col] = st.number_input(f"{col}", value=val)

for col in categorical_cols:
    le = le_dict[col]
    options = list(le.classes_)
    selected = st.selectbox(f"{col}", options)
    input_data[col] = le.transform([selected])[0]

for col in ['Year', 'Month']:
    val = int(df[col].mode()[0])
    input_data[col] = st.number_input(f"{col}", value=val)

input_df = pd.DataFrame([input_data])

input_df[numeric_cols] = pt.transform(input_df[numeric_cols])
input_scaled = scaler.transform(input_df[features])
input_pca = pca.transform(input_scaled)

st.subheader("⚙️ Select Clustering Model")
model_choice = st.selectbox("Choose Model", ["KMeans", "Hierarchical", "DBSCAN"])

if st.button("🔍 Predict & Show Metrics"):

    if model_choice == "KMeans":
        labels = kmeans.labels_
        cluster_label = kmeans.predict(input_pca)[0]

    elif model_choice == "Hierarchical":
        labels = fcluster(Z, t=3, criterion='maxclust')
        cluster_label = labels[np.argmin(np.linalg.norm(X_pca - input_pca, axis=1))]

    elif model_choice == "DBSCAN":
        labels = dbscan.labels_
        core_mask = labels != -1
        if core_mask.any():
            core_points = X_pca[core_mask]
            core_labels = labels[core_mask]
            nearest_idx = np.argmin(np.linalg.norm(core_points - input_pca, axis=1))
            cluster_label = core_labels[nearest_idx]
        else:
            cluster_label = -1

    valid_mask = labels != -1
    X_valid = X_pca[valid_mask]
    labels_valid = labels[valid_mask]

    try:
        sil = silhouette_score(X_valid, labels_valid)
        dbi = davies_bouldin_score(X_valid, labels_valid)
        ch = calinski_harabasz_score(X_valid, labels_valid)
    except:
        sil, dbi, ch = np.nan, np.nan, np.nan

    st.markdown(f"### 🏷️ Cluster Assigned: `{cluster_label}`")
    st.write(f"**Silhouette Score:** {sil:.4f}")
    st.write(f"**Davies–Bouldin Index:** {dbi:.4f}")
    st.write(f"**Calinski–Harabasz Score:** {ch:.4f}")

    st.subheader("🌀 PCA Cluster Visualization")
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='tab10', s=50)
    plt.scatter(input_pca[:,0], input_pca[:,1], color='red', s=200, marker='X', label='New Point')
    plt.legend()
    plt.title(f"PCA Visualization ({model_choice})")
    st.pyplot(plt.gcf())
