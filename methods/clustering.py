from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class Clustering:
    @staticmethod
    def kmeans_clustering(X, n_clusters=10, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_labels = kmeans.fit_predict(X)
        return kmeans_labels

    @staticmethod
    def DBSCAN_clustering(X, eps=0.5, minpts=5):
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)
        print(X_scaled)

        dbscan = DBSCAN(eps=eps, min_samples=minpts)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        return dbscan_labels

