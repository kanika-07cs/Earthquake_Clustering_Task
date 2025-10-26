import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster

import mlflow
import mlflow.sklearn

df = pd.read_csv("database.csv")
df.dropna(axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%Y')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce', format='%H:%M:%S')
df.dropna(subset=['Date', 'Time'], inplace=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Hour'] = df['Time'].dt.hour
df['Min'] = df['Time'].dt.minute

label_cols = ['Type', 'Source', 'Location Source', 'Magnitude Source', 'Status']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

numeric_cols = ['Latitude', 'Longitude', 'Magnitude', 'Depth']
pt = PowerTransformer(method='yeo-johnson')
df[numeric_cols] = pt.fit_transform(df[numeric_cols])
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

features = ['Latitude', 'Longitude', 'Depth', 'Magnitude',
            'Type', 'Source', 'Magnitude Source', 'Status',
            'Year', 'Month']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

Z = linkage(X_pca, method='ward')

mlflow.set_experiment("Earthquake_Clustering")

def plot_and_log(df_plot, cluster_col, title, filename):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue=cluster_col, data=df_plot, palette='Set2', s=50)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Cluster')
    plt.savefig(filename)
    plt.close()
    mlflow.log_artifact(filename)

with mlflow.start_run(run_name="KMeans"):
    n_clusters_km = 3
    kmeans = KMeans(n_clusters=n_clusters_km, random_state=42)
    km_labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, km_labels)
    dbi = davies_bouldin_score(X_pca, km_labels)
    ch = calinski_harabasz_score(X_pca, km_labels)

    mlflow.sklearn.log_model(kmeans, "kmeans_model")
    mlflow.log_param("n_clusters", n_clusters_km)
    mlflow.log_metric("silhouette", sil)
    mlflow.log_metric("davies_bouldin", dbi)
    mlflow.log_metric("calinski_harabasz", ch)

    pca_df['KMeans_Cluster'] = km_labels
    pca_df.to_csv("kmeans_clustered.csv", index=False)
    mlflow.log_artifact("artifacts/kmeans_clustered.csv")

    plot_and_log(pca_df, 'KMeans_Cluster', 'KMeans Clustering (2D PCA)', 'artifacts/kmeans_cluster_plot.png')

with mlflow.start_run(run_name="Hierarchical"):
    hc_labels = fcluster(Z, t=3, criterion='maxclust')

    sil = silhouette_score(X_pca, hc_labels)
    dbi = davies_bouldin_score(X_pca, hc_labels)
    ch = calinski_harabasz_score(X_pca, hc_labels)

    mlflow.log_param("n_clusters", 3)
    mlflow.log_metric("silhouette", sil)
    mlflow.log_metric("davies_bouldin", dbi)
    mlflow.log_metric("calinski_harabasz", ch)

    pca_df['Hier_Cluster'] = hc_labels
    pca_df.to_csv("hierarchical_clustered.csv", index=False)
    mlflow.log_artifact("hierarchical_clustered.csv")

    plot_and_log(pca_df, 'Hier_Cluster', 'Hierarchical Clustering (2D PCA)', 'artifacts/hierarchical_cluster_plot.png')

with mlflow.start_run(run_name="DBSCAN"):
    eps_val = 1.2
    min_samples_val = 4
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    db_labels = dbscan.fit_predict(X_pca)

    mask = db_labels != -1
    if len(set(db_labels[mask])) > 1:
        sil = silhouette_score(X_pca[mask], db_labels[mask])
        dbi = davies_bouldin_score(X_pca[mask], db_labels[mask])
        ch = calinski_harabasz_score(X_pca[mask], db_labels[mask])
    else:
        sil, dbi, ch = 0, 0, 0

    mlflow.sklearn.log_model(dbscan, "dbscan_model")
    mlflow.log_param("eps", eps_val)
    mlflow.log_param("min_samples", min_samples_val)
    mlflow.log_metric("silhouette", sil)
    mlflow.log_metric("davies_bouldin", dbi)
    mlflow.log_metric("calinski_harabasz", ch)

    pca_df['DBSCAN_Cluster'] = db_labels
    pca_df.to_csv("dbscan_clustered.csv", index=False)
    mlflow.log_artifact("artifacts/dbscan_clustered.csv")

    plot_and_log(pca_df, 'DBSCAN_Cluster', 'DBSCAN Clustering (2D PCA)', 'artifacts/dbscan_cluster_plot.png')

print("Separate MLflow runs completed with cluster plots and models logged!")
