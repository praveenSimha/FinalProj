import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_distribution(data=None, data_path=None, output_path=None):
    """
    Generates a bar plot showing the distribution of customers in each cluster.
    Accepts either a DataFrame or a file path.
    """
    if data is None and data_path is not None:
        data = pd.read_csv(data_path)
    elif data is None:
        print("No data provided for cluster distribution plot.")
        return

    if 'Cluster' not in data.columns:
        print("Cluster column not found in the data.")
        return

    # Plot the cluster distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=data)
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}/cluster_distribution.png")
        plt.close()
    else:
        plt.show()

def plot_spending_trends(data=None, data_path=None, output_path=None):
    """
    Generates a bar plot showing the average spending trends per cluster.
    Accepts either a DataFrame or a file path.
    """
    if data is None and data_path is not None:
        data = pd.read_csv(data_path)
    elif data is None:
        print("No data provided for spending trends plot.")
        return

    if 'Cluster' not in data.columns or 'PURCHASES' not in data.columns:
        print("Required columns ('Cluster', 'PURCHASES') not found in the data.")
        return

    # Group by cluster and calculate the average purchase value
    spending_data = data.groupby('Cluster')['PURCHASES'].mean()

    # Plot the spending trends
    plt.figure(figsize=(8, 6))
    spending_data.plot(kind='bar')
    plt.title('Average Spending by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Spending')

    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}/spending_trends.png")
        plt.close()
    else:
        plt.show()
