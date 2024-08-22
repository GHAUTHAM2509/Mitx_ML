import numpy as np

# Dataset
data = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])

# Initial centers for k-means and k-medoids
initial_centers = np.array([[-5, 2], [0, -6]])

# Function for K-means clustering
def k_means(data, k, initial_centers, max_iter=100):
    centers = initial_centers.copy()
    for _ in range(max_iter):
        # Assignment step
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centers, axis=2), axis=1)
        # Update step
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# Function for K-medoids clustering using L1 norm
def k_medoids(data, k, initial_centers, max_iter=100):
    def manhattan_distance(x, y):
        return np.sum(np.abs(x - y))
        # return np.linalg.norm(x - y, ord=1)

    centers = initial_centers.copy()
    labels = np.zeros(data.shape[0])
    for _ in range(max_iter):
        # Assignment step
        for i in range(data.shape[0]):
            distances = np.array([manhattan_distance(data[i], center) for center in centers])
            labels[i] = np.argmin(distances)
        # Update step
        new_centers = np.copy(centers)
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) == 0:
                continue
            medoid_costs = np.array([np.sum([manhattan_distance(point, other) for other in cluster_points])
                                     for point in cluster_points])
            new_centers[j] = cluster_points[np.argmin(medoid_costs)]
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# Performing K-means clustering
k_means_centers, k_means_labels = k_means(data, 2, initial_centers)
print("K-means Centers:", k_means_centers)
print("K-means Labels:", k_means_labels)

# Performing K-medoids clustering
k_medoids_centers, k_medoids_labels = k_medoids(data, 2, initial_centers)
print("K-medoids Centers:", k_medoids_centers)
print("K-medoids Labels:", k_medoids_labels)

# Format output for clusters and centers as requested
def format_clusters(centers, labels, data):
    clusters = []
    for i, center in enumerate(centers):
        cluster_points = data[labels == i]
        clusters.append(f"[{int(center[0])}, {int(center[1])}]; {'; '.join(map(str, cluster_points.tolist()))}")
    return clusters

# Printing formatted clusters and centers for K-means
print("\nK-means Clusters and Centers:")
for cluster in format_clusters(k_means_centers, k_means_labels, data):
    print(cluster)

# Printing formatted clusters and centers for K-medoids
print("\nK-medoids Clusters and Centers:")
for cluster in format_clusters(k_medoids_centers, k_medoids_labels, data):
    print(cluster)