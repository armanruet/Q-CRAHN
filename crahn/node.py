import numpy as np
from collections import defaultdict
import random


class Node:
    """Base class for network nodes"""

    def __init__(self, node_id, position):
        self.id = node_id
        self.position = position  # (x, y)

    def distance_to(self, other_node):
        """Calculate Euclidean distance to another node"""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(self.position, other_node.position)))


class CRAHNNode(Node):
    """Secondary User (SU) node in CRAHN"""

    def __init__(self, node_id, position, initial_energy=None):
        super().__init__(node_id, position)
        self.initial_energy = initial_energy if initial_energy else random.uniform(
            0.7, 1.5)
        self.residual_energy = self.initial_energy
        self.available_channels = set()
        self.neighbors = set()
        self.q_values = defaultdict(float)
        self.is_ch = False
        self.cluster_id = None
        self.role = 'undefined'  # 'CH', 'MN', 'GN'

    def update_q_value(self, channel, reward, learning_rate, discount_factor, max_next_q):
        """Update Q-value for a channel using Q-learning update rule"""
        self.q_values[channel] = (1 - learning_rate) * self.q_values[channel] + \
            learning_rate * (reward + discount_factor * max_next_q)

    def get_channel_fitness(self):
        """Calculate channel fitness value based on Q-values and neighbor connectivity"""
        fitness = 0
        for channel in self.available_channels:
            connectable_neighbors = len(
                self.get_connectable_neighbors(channel))
            fitness += self.q_values[channel] * connectable_neighbors
        return fitness

    def get_connectable_neighbors(self, channel):
        """Get list of neighbors that can communicate on given channel"""
        return [n for n in self.neighbors if channel in n.available_channels]

    def consume_energy(self, amount):
        """Consume energy and return True if node is still alive"""
        self.residual_energy -= amount
        return self.residual_energy > 0


class PrimaryUser(Node):
    """Primary User (PU) node in CRAHN"""

    def __init__(self, pu_id, position, channel, range=50):
        super().__init__(pu_id, position)
        self.channel = channel
        self.range = range
        self.is_active = True

    def affects_node(self, node):
        """Check if this PU affects a given node"""
        return self.is_active and self.distance_to(node) <= self.range
