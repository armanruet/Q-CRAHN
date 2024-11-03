import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from .node import CRAHNNode, PrimaryUser
from .utils import plot_network_topology, plot_performance_metrics, calculate_network_metrics


class CRAHNSimulation:
    """
    Cognitive Radio Ad Hoc Network (CRAHN) Simulator implementing Q-learning based clustering
    """

    def __init__(
        self,
        area_size: int = 100,
        n_nodes: int = 40,
        n_pus: int = 12,
        n_channels: int = 12,
        transmission_range: float = 30.0,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        ch_selection_threshold: float = 0.5,
        energy_weight: float = 0.25,
        channel_weight: float = 0.25,
        neighbor_weight: float = 0.25,
        cluster_weight: float = 0.25
    ):
        """
        Initialize CRAHN simulation environment

        Args:
            area_size: Simulation area size (square)
            n_nodes: Number of secondary user nodes
            n_pus: Number of primary users
            n_channels: Number of available channels
            transmission_range: Node transmission range
            learning_rate: Q-learning rate (α)
            discount_factor: Q-learning discount factor (γ)
            ch_selection_threshold: Threshold for CH selection
            energy_weight: Weight for energy in fitness calculation (β1)
            channel_weight: Weight for channel quality in fitness calculation (β2)
            neighbor_weight: Weight for neighbor count in fitness calculation (β3)
            cluster_weight: Weight for cluster reachability in fitness calculation (β4)
        """
        # Initialize simulation parameters
        self.area_size = area_size
        self.n_nodes = n_nodes
        self.n_pus = n_pus
        self.n_channels = n_channels
        self.transmission_range = transmission_range
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.ch_selection_threshold = ch_selection_threshold

        # Weights for multi-objective optimization
        self.weights = {
            'energy': energy_weight,
            'channel': channel_weight,
            'neighbor': neighbor_weight,
            'cluster': cluster_weight
        }

        # Initialize network components
        self.nodes: List[CRAHNNode] = self._initialize_nodes()
        self.pus: List[PrimaryUser] = self._initialize_pus()
        self.clusters: Dict = {}

        # Performance metrics storage
        self.metrics = defaultdict(list)

    def _initialize_nodes(self) -> List[CRAHNNode]:
        """Initialize secondary user nodes with random positions"""
        return [
            CRAHNNode(
                node_id=i,
                position=(
                    random.uniform(0, self.area_size),
                    random.uniform(0, self.area_size)
                )
            )
            for i in range(self.n_nodes)
        ]

    def _initialize_pus(self) -> List[PrimaryUser]:
        """Initialize primary users with random positions and channel assignments"""
        return [
            PrimaryUser(
                pu_id=i,
                position=(
                    random.uniform(0, self.area_size),
                    random.uniform(0, self.area_size)
                ),
                channel=i % self.n_channels
            )
            for i in range(self.n_pus)
        ]

    def update_available_channels(self):
        """Update available channels for each node based on PU activity"""
        for node in self.nodes:
            # Initially all channels are available
            node.available_channels = set(range(self.n_channels))

            # Remove channels based on PU proximity
            for pu in self.pus:
                if pu.affects_node(node):
                    node.available_channels.discard(pu.channel)

    def update_neighbors(self):
        """Update neighbor lists for all nodes based on transmission range"""
        for node in self.nodes:
            node.neighbors.clear()
            for other_node in self.nodes:
                if node != other_node and node.distance_to(other_node) <= self.transmission_range:
                    node.neighbors.add(other_node.id)

    def calculate_channel_reward(self, node: CRAHNNode, channel: int) -> float:
        """
        Calculate reward for a channel based on availability and quality

        Args:
            node: Secondary user node
            channel: Channel number

        Returns:
            float: Calculated reward value
        """
        if channel in node.available_channels:
            # Simulate channel conditions
            idle_prob = random.uniform(0.6, 1.0)  # Channel idle probability
            # Normalized average idle time
            avg_idle_time = random.uniform(0.5, 1.0)

            # Weighted sum of channel metrics (ω1 and ω2 from paper)
            return 0.5 * avg_idle_time + 0.5 * idle_prob
        return 0.0

    def update_q_values(self):
        """Update Q-values for all nodes using Q-learning"""
        for node in self.nodes:
            for channel in range(self.n_channels):
                reward = self.calculate_channel_reward(node, channel)
                max_next_q = max(node.q_values[ch]
                                 for ch in range(self.n_channels))

                # Q-learning update rule from equation (1) in paper
                node.update_q_value(
                    channel=channel,
                    reward=reward,
                    learning_rate=self.learning_rate,
                    discount_factor=self.discount_factor,
                    max_next_q=max_next_q
                )

    def calculate_channel_fitness(self, node: CRAHNNode) -> float:
        """
        Calculate channel fitness value (CF) for a node using equation (8) from paper

        Args:
            node: Node to evaluate channel fitness

        Returns:
            float: Channel fitness value
        """
        fitness = 0
        effective_channels = node.available_channels - set(
            nc.cadc for nc in self.get_neighbor_clusters(node)
        )

        for channel in effective_channels:
            connectable_neighbors = len(
                node.get_connectable_neighbors(channel))
            fitness += node.q_values[channel] * connectable_neighbors

        return fitness

    def calculate_ch_fitness(self, node: CRAHNNode) -> float:
        """
        Calculate cluster head fitness value using equation (11) from paper

        Args:
            node: Node to evaluate for CH fitness

        Returns:
            float: CH fitness value
        """
        # Channel fitness
        channel_fitness = self.calculate_channel_fitness(node)

        # Calculate normalized components
        energy_component = node.residual_energy / 1.5  # Max energy is 1.5J
        channel_component = channel_fitness / (self.n_channels * self.n_nodes)
        neighbor_component = len(node.neighbors) / self.n_nodes

        # Number of reachable neighbor clusters
        reachable_clusters = len(self.get_neighbor_clusters(node))
        cluster_component = reachable_clusters / \
            len(self.clusters) if self.clusters else 0

        # Weighted sum using β weights from paper
        return (
            self.weights['energy'] * energy_component +
            self.weights['channel'] * channel_component +
            self.weights['neighbor'] * neighbor_component +
            self.weights['cluster'] * cluster_component
        )

    def get_neighbor_clusters(self, node: CRAHNNode) -> Set:
        """Get set of neighbor clusters for a node"""
        neighbor_clusters = set()
        for neighbor_id in node.neighbors:
            neighbor = self.nodes[neighbor_id]
            if neighbor.cluster_id is not None:
                neighbor_clusters.add(self.clusters[neighbor.cluster_id])
        return neighbor_clusters

    def select_cluster_heads(self):
        """Select cluster heads based on fitness values"""
        self.clusters.clear()
        candidates = []

        # Calculate fitness values and select candidates
        for node in self.nodes:
            fitness = self.calculate_ch_fitness(node)
            if fitness > self.ch_selection_threshold:
                candidates.append((node.id, fitness))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Select CHs and assign members
        for ch_id, _ in candidates:
            if not self.nodes[ch_id].cluster_id:  # If node not already in a cluster
                ch_node = self.nodes[ch_id]
                ch_node.is_ch = True
                ch_node.role = 'CH'
                ch_node.cluster_id = ch_id

                # Create new cluster
                self.clusters[ch_id] = {
                    'CH': ch_id,
                    'MNs': set(),
                    'GNs': set(),
                    'cadc': self.select_common_channel(ch_node),
                    'cbdc': None  # Will be set later
                }

                # Assign member nodes
                self.assign_member_nodes(ch_node)

    def select_common_channel(self, ch_node: CRAHNNode) -> int:
        """
        Select common active data channel (CADC) using equation (15) from paper

        Args:
            ch_node: Cluster head node

        Returns:
            int: Selected channel number
        """
        best_channel = None
        best_value = -1

        for channel in ch_node.available_channels:
            connectable_nodes = len(ch_node.get_connectable_neighbors(channel))
            value = ch_node.q_values[channel] * connectable_nodes

            if value > best_value:
                best_value = value
                best_channel = channel

        return best_channel

    def assign_member_nodes(self, ch_node: CRAHNNode):
        """
        Assign member nodes to a cluster

        Args:
            ch_node: Cluster head node
        """
        cluster = self.clusters[ch_node.id]
        cadc = cluster['cadc']

        for neighbor_id in ch_node.neighbors:
            neighbor = self.nodes[neighbor_id]
            if (not neighbor.cluster_id and  # Not already in a cluster
                    cadc in neighbor.available_channels):  # Can use cluster's CADC
                neighbor.cluster_id = ch_node.id
                neighbor.role = 'MN'
                cluster['MNs'].add(neighbor_id)

    def select_gateway_nodes(self):
        """Select gateway nodes for inter-cluster communication using equation (17)"""
        for cluster_id, cluster in self.clusters.items():
            ch_node = self.nodes[cluster_id]
            neighbor_clusters = self.get_neighbor_clusters(ch_node)

            for nc in neighbor_clusters:
                best_gn = None
                best_q_value = -1

                # Find best gateway node candidate
                for mn_id in cluster['MNs']:
                    mn_node = self.nodes[mn_id]
                    if (cluster['cadc'] in mn_node.available_channels and
                            nc['cadc'] in mn_node.available_channels):
                        # Average Q-value for both channels
                        avg_q = (mn_node.q_values[cluster['cadc']] +
                                 mn_node.q_values[nc['cadc']]) / 2

                        if avg_q > best_q_value:
                            best_q_value = avg_q
                            best_gn = mn_id

                # Assign gateway node
                if best_gn is not None:
                    self.nodes[best_gn].role = 'GN'
                    cluster['GNs'].add(best_gn)

    def run_simulation(self, n_iterations: int = 100) -> Dict:
        """
        Run the CRAHN simulation

        Args:
            n_iterations: Number of simulation iterations

        Returns:
            Dict: Performance metrics
        """
        for iteration in range(n_iterations):
            # Update network state
            self.update_available_channels()
            self.update_neighbors()
            self.update_q_values()

            # Form clusters
            self.select_cluster_heads()
            self.select_gateway_nodes()

            # Calculate and store metrics
            metrics = calculate_network_metrics(self.nodes, self.clusters)
            for key, value in metrics.items():
                self.metrics[key].append(value)

            # Simulate energy consumption
            for node in self.nodes:
                if node.is_ch:
                    # Higher energy consumption for CHs
                    node.consume_energy(0.01)
                else:
                    # Lower energy consumption for MNs
                    node.consume_energy(0.005)

        return dict(self.metrics)

    def visualize_results(self):
        """Visualize simulation results"""
        # Plot network topology
        topology_fig = plot_network_topology(
            nodes=self.nodes,
            pus=self.pus,
            clusters=self.clusters,
            area_size=self.area_size
        )
        topology_fig.savefig('network_topology.png')

        # Plot performance metrics
        metrics_fig = plot_performance_metrics(self.metrics)
        metrics_fig.savefig('performance_metrics.png')

        plt.close('all')

    def get_simulation_summary(self) -> Dict:
        """
        Get summary of simulation results

        Returns:
            Dict: Summary statistics
        """
        return {
            'n_clusters': len(self.clusters),
            'avg_cluster_size': np.mean([len(c['MNs']) for c in self.clusters.values()]),
            'avg_ch_energy': np.mean([self.nodes[ch_id].residual_energy
                                      for ch_id in self.clusters.keys()]),
            'n_gateway_nodes': sum(len(c['GNs']) for c in self.clusters.values()),
            'active_nodes': sum(1 for node in self.nodes if node.residual_energy > 0),
            'total_energy': sum(node.residual_energy for node in self.nodes)
        }

    def switch_to_backup_channel(self, cluster_id):
        """Switch cluster from CADC to CBDC when primary user appears"""
        cluster = self.clusters[cluster_id]
        if cluster['cbdc'] is not None:
            temp = cluster['cadc']
            cluster['cadc'] = cluster['cbdc']
            cluster['cbdc'] = temp
            return True
        return False

    def remove_member_node(self, node_id, cluster_id):
        """Remove a member node from cluster"""
        if cluster_id in self.clusters:
            cluster = self.clusters[cluster_id]
            if node_id in cluster['MNs']:
                cluster['MNs'].remove(node_id)
                if node_id in cluster['GNs']:
                    cluster['GNs'].remove(node_id)
                self.nodes[node_id].cluster_id = None
                self.nodes[node_id].role = 'undefined'
                return True
        return False


def add_member_node(self, node_id, cluster_id):
    """Add a new member node to cluster"""
    if cluster_id in self.clusters:
        node = self.nodes[node_id]
        cluster = self.clusters[cluster_id]

        if (cluster['cadc'] in node.available_channels and
                node.distance_to(self.nodes[cluster_id]) <= self.transmission_range):
            cluster['MNs'].add(node_id)
            node.cluster_id = cluster_id
            node.role = 'MN'
            return True
    return False
