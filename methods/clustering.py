from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes


class Clustering:
    @staticmethod
    def kmeans_clustering(X, n_clusters=10, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_labels = kmeans.fit_predict(X)
        return kmeans_labels

    @staticmethod
    def DBSCAN_clustering(X, eps=0.5, minpts=5):
        scaler = StandardScaler(with_mean=False)
        # scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        print(X_scaled)

        dbscan = DBSCAN(eps=eps, min_samples=minpts)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        return dbscan_labels
    
    @staticmethod
    def kprototypes_clustering(X, dn, n_clusters=10, random_state=42):
        kproto = KPrototypes(n_clusters=n_clusters, random_state=random_state)
        categorical_indices = list(range(dn, X.shape[1]))

        kproto_labels = kproto.fit_predict(X, categorical=categorical_indices)
        return kproto_labels
    
    @staticmethod
    def kmodes_clustering(X, n_clusters=10, random_state=42):
        kmodes = KModes(n_clusters=n_clusters, init='Huang', n_init=10, random_state=random_state)
        
        kmodes_labels = kmodes.fit_predict(X)
        return kmodes_labels


