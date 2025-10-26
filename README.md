# 🌎 Earthquake Clustering Dashboard

## Project Overview
This project is focused on analyzing earthquake data using clustering techniques. The goal is to identify patterns and categorize regions based on seismic activity. The dashboard allows interactive 
exploration of clustering results and metrics for new earthquake events using Streamlit. MLflow is integrated to track experiments, parameters, metrics, and artifacts for reproducibility.

The clustering models used include:
- **KMeans**: Groups earthquakes into clusters based on similarity.
- **Hierarchical Clustering**: Groups earthquakes using agglomerative clustering.
- **DBSCAN**: Detects high-density clusters and identifies outliers (noise).

**Dataset link**: https://www.kaggle.com/datasets/usgs/earthquake-database

**Streamlit Link**: https://earthquakeclusteringtask-c3eayj7cjjpe3hsrszrcab.streamlit.app/

## Dataset Description
- **Latitude, Longitude**: Geographic coordinates.
- **Depth**: Depth of the earthquake (km).
- **Magnitude**: Magnitude of the earthquake.
- **Type**: Earthquake type (e.g., tectonic, volcanic).
- **Source, Location Source, Magnitude Source**: Reporting sources.
- **Status**: Event status.
- **Date, Time**: Date and time of the earthquake.

## Data Preprocessing Steps
1. **Missing Value Handling**: Drop columns and rows with missing critical values.
2. **Date/Time Conversion**: Converted `Date` and `Time` to datetime objects and extracted `Year`, `Month`, `Hour`, `Minute`.
3. **Label Encoding**: Converted categorical variables (`Type`, `Source`, `Location Source`, `Magnitude Source`, `Status`) to numeric values using `LabelEncoder`.
4. **Power Transformation**: Applied Yeo-Johnson transformation on numeric columns (`Latitude`, `Longitude`, `Depth`, `Magnitude`) to reduce skewness.
5. **Outlier Removal**: Used IQR method to clip outliers in numeric features.
6. **Scaling & PCA**: Standardized features using `StandardScaler` and reduced dimensions to 2 using PCA for visualization.

## Model Development
**1. KMeans**
- `n_clusters = 3`
- Groups earthquakes based on PCA-reduced features.
- Provides clear cluster labels for regions.
**2. Hierarchical Clustering**
- Agglomerative clustering with `ward` linkage.
- Produces a dendrogram for visual analysis.
- Cluster assignment based on maximum clusters (`t=3`).
**3. DBSCAN**
- Density-based clustering (`eps=1.2`, `min_samples=4`).
- Identifies dense clusters and noise points (outliers).
- Works well for non-globular cluster shapes.

All models were trained on preprocessed PCA-transformed data and saved using `joblib`.

## MLflow Integration
MLflow is used to track experiments, parameters, metrics, and artifacts:
- **Parameters tracked**: Model type, hyperparameters (e.g., `n_clusters`, `eps`, `min_samples`).
- **Metrics tracked**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score.
- **Artifacts tracked**: Saved model files (`*.pkl`), PCA components, plots.

## Evaluation
Clustering performance was evaluated using:
- **Silhouette Score**: Measures cluster cohesion and separation.
- **Davies-Bouldin Index (DBI)**: Lower values indicate better clustering.
- **Calinski-Harabasz Score (CH)**: Higher values indicate dense and well-separated clusters.

Metrics were calculated for each clustering method and displayed on the dashboard.

## Results & Insights
- **KMeans** separated earthquake events into three meaningful zones:
  1. Strong earthquake zones.
  2. Moderate/Calmer zones.
  3. Minor/Inactive zones.

- **Hierarchical Clustering** provided a dendrogram for visualizing relationships between events.
- **DBSCAN** effectively detected high-density seismic regions and noise/outliers.

## How to Run
1. Clone the repository:
- git clone <repository_url>
- cd Earthquake_Clustering_Task
2. Ensure all .pkl models are inside the models/ folder.
3. Run the Streamlit app:
- streamlit run app.py
4. Access MLflow to monitor experiments
  - mlflow ui

## Conclusion
This project demonstrates clustering-based analysis of earthquake data. By combining preprocessing, PCA, multiple clustering algorithms, and MLflow tracking, it ensures reproducible, interpretable results. 
The Streamlit dashboard provides an interactive interface for exploring historical data and predicting cluster assignments for new events.
