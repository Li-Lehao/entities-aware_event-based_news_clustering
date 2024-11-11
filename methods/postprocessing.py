import numpy as np
from sklearn.metrics import jaccard_score

def post_processing(X_embedding_entity, labels, similarity_threshold=0.5):
    """
    Removes dissimilar items within each cluster based on Jaccard distance.
    
    Parameters:
        X_embedding_entity (np.ndarray): A 2D array where each row is a vector representing an entity's embedding.
        labels (np.ndarray): An array of clustering labels corresponding to each vector.
        similarity_threshold (float): The minimum average similarity score required for a vector to stay in the cluster.
    
    Returns:
        np.ndarray: Updated labels with dissimilar items removed, marked as -1.
    """
    # Convert input to numpy arrays if not already
    X_embedding_entity = np.asarray(X_embedding_entity)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    
    # Copy labels to allow modifications
    updated_labels = labels.copy()
    
    for cluster in unique_labels:
        if cluster == -1:  # Ignore noise label if it exists
            continue
            
        # Get indices of vectors in the current cluster
        cluster_indices = np.where(labels == cluster)[0]
        cluster_vectors = X_embedding_entity[cluster_indices]
        
        # Calculate pairwise Jaccard distances within the cluster
        dissimilar_items = []
        for i, vector in enumerate(cluster_vectors):
            similarities = []
            for j, other_vector in enumerate(cluster_vectors):
                if i != j:
                    # Compute Jaccard similarity
                    similarity = jaccard_score(vector, other_vector, average='binary')
                    similarities.append(similarity)
                    
            # Check if the average similarity meets the threshold
            if np.mean(similarities) < similarity_threshold:
                dissimilar_items.append(cluster_indices[i])
        
        # Mark dissimilar items with label -1 (indicating noise or exclusion)
        updated_labels[dissimilar_items] = -1
        
    return updated_labels
