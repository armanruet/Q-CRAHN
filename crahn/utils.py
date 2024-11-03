import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_network_topology(nodes, pus, clusters, area_size):
    """Plot network topology with nodes, PUs, and clusters"""
    plt.figure(figsize=(10, 10))

    # Plot PUs and their ranges
    for pu in pus:
        plt.plot(pu.position[0], pu.position[1], 'r^', markersize=10,
                 label='PU' if pu == pus[0] else "")
        circle = Circle(pu.position, pu.range,
                        color='r', fill=False, alpha=0.2)
        plt.gca().add_patch(circle)

    # Plot nodes with role-based colors
    colors = {'CH': 'g', 'MN': 'b', 'GN': 'y', 'undefined': 'k'}
    for node in nodes:
        plt.plot(node.position[0], node.position[1], f"{colors[node.role]}o",
                 markersize=8, label=node.role if node.role not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot cluster connections
    for cluster_id, cluster in clusters.items():
        ch_pos = nodes[cluster['CH']].position
        for mn_id in cluster['MNs']:
            mn_pos = nodes[mn_id].position
            plt.plot([ch_pos[0], mn_pos[0]], [
                     ch_pos[1], mn_pos[1]], 'k--', alpha=0.3)

    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.grid(True)
    plt.legend()
    plt.title('CRAHN Network Topology')

    return plt.gcf()


def plot_performance_metrics(metrics):
    """Plot performance metrics over time"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Average cluster size
    axes[0].plot(metrics['avg_cluster_size'])
    axes[0].set_title('Average Cluster Size')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Nodes per Cluster')

    # Average CH energy
    axes[1].plot(metrics['avg_ch_energy'])
    axes[1].set_title('Average CH Energy')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Energy (J)')

    # Number of clusters
    axes[2].plot(metrics['n_clusters'])
    axes[2].set_title('Number of Clusters')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Clusters')

    plt.tight_layout()
    return fig


def calculate_network_metrics(nodes, clusters):
    """Calculate various network performance metrics"""
    metrics = {
        'avg_cluster_size': np.mean([len(cluster['MNs']) for cluster in clusters.values()]) if clusters else 0,
        'avg_ch_energy': np.mean([nodes[ch_id].residual_energy for ch_id in clusters.keys()]) if clusters else 0,
        'n_clusters': len(clusters),
        'total_energy': sum(node.residual_energy for node in nodes),
        'active_nodes': sum(1 for node in nodes if node.residual_energy > 0)
    }
    return metrics


def save_metrics(metrics, filename):
    """Save performance metrics to file"""
    import json
    with open(filename, 'w') as f:
        json.dump(metrics, f)


def load_metrics(filename):
    """Load performance metrics from file"""
    import json
    with open(filename, 'r') as f:
        return json.load(f)
