from sklearn.metrics import silhouette_score

def evaluate_clustering(data, cluster_column='Cluster'):
    """
    Evaluates the clustering performance using silhouette score.
    Handles cases where silhouette score cannot be calculated due to insufficient data.
    """
    numerical_data = data.select_dtypes(include='number')
    
    # Check if we have enough data to calculate silhouette score
    if len(data) < 2 or len(data[cluster_column].unique()) < 2:
        print("Not enough data points or clusters to calculate Silhouette Score.")
        return None

    # Calculate silhouette score
    score = silhouette_score(numerical_data, data[cluster_column])
    return score

def print_clustering_summary(data):
    """
    Prints a summary of the clusters, including their size and average spending.
    """
    if 'Cluster' not in data.columns:
        print("Cluster column not found in the data.")
        return
    
    print("Clustering Summary:")
    cluster_summary = data.groupby('Cluster').agg(
        customer_count=('Cluster', 'size'),
        avg_spending=('PURCHASES', 'mean')
    )
    print(cluster_summary)
