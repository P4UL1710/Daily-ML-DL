# Mall Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def main():
    # Load dataset
    data = pd.read_csv("Mall_Customers.csv")

    print("First 5 Rows:")
    print(data.head())

    print("\nDataset Shape:")
    print(data.shape)

    print("\nDataset Information:")
    print(data.info())

    print("\nMissing Values:")
    print(data.isnull().sum())

    # Selecting Annual Income and Spending Score columns
    X = data.iloc[:, [3, 4]].values

    # Finding WCSS values for different numbers of clusters
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(
            n_clusters=i,
            init="k-means++",
            random_state=42,
            n_init=10
        )
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Elbow Method Graph
    sns.set()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title("The Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    # Training K-Means Model
    kmeans = KMeans(
        n_clusters=5,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    Y = kmeans.fit_predict(X)

    # Visualizing Clusters
    plt.figure(figsize=(8, 8))

    plt.scatter(X[Y == 0, 0], X[Y == 0, 1],
                s=50, c='green', label='Cluster 1')

    plt.scatter(X[Y == 1, 0], X[Y == 1, 1],
                s=50, c='red', label='Cluster 2')

    plt.scatter(X[Y == 2, 0], X[Y == 2, 1],
                s=50, c='yellow', label='Cluster 3')

    plt.scatter(X[Y == 3, 0], X[Y == 3, 1],
                s=50, c='blue', label='Cluster 4')

    plt.scatter(X[Y == 4, 0], X[Y == 4, 1],
                s=50, c='violet', label='Cluster 5')

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        c='black',
        marker='X',
        label='Centroids'
    )

    plt.title("Customer Groups")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()