# Q-Learning Based Cognitive Radio Network Simulator

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2019.2959313-b31b1b.svg)](https://www.dropbox.com/scl/fi/vo5x3hons46fywgq0ieih/08932525.pdf?rlkey=g52l6mb038100vug5x826hsih&e=1&dl=0)

This repository imitates a simple version of the Q-learning based clustering algorithm for Cognitive Radio Ad Hoc Networks (CRAHN) as described in the paper "Q-Learning Based Multi-Objective Clustering Algorithm for Cognitive Radio Ad Hoc Networks" (IEEE Access, 2019).

## 📖 Overview

### System Architecture

```mermaid
flowchart TB
    subgraph "Network Layer"
        N1[Node Discovery]
        N2[Channel Management]
        N3[Topology Control]
    end
    
    subgraph "Learning Layer"
        L1[Q-Learning Module]
        L2[Channel Quality Assessment]
        L3[Reward Calculation]
    end
    
    subgraph "Clustering Layer"
        C1[Cluster Head Selection]
        C2[Member Assignment]
        C3[Gateway Selection]
    end
    
    N1 --> L1
    N2 --> L2
    L1 --> L3
    L2 --> L3
    L3 --> C1
    C1 --> C2
    C2 --> C3
    N3 --> C1
```

### Use Case Scenarios

```mermaid
graph TB
    subgraph "Primary Users"
        PU1[Channel Occupancy]
        PU2[Channel Release]
    end
    
    subgraph "Secondary Users"
        SU1[Spectrum Sensing]
        SU2[Channel Selection]
        SU3[Cluster Formation]
        SU4[Data Transmission]
    end
    
    PU1 --> SU1
    PU2 --> SU1
    SU1 --> SU2
    SU2 --> SU3
    SU3 --> SU4
```

### Node State Machine

```mermaid
stateDiagram-v2
    [*] --> Initialization
    Initialization --> ChannelSensing: Start
    ChannelSensing --> QValueUpdate: Channel Quality
    QValueUpdate --> ClusterFormation: Q-Values Ready
    ClusterFormation --> MemberNode: Join Request
    ClusterFormation --> ClusterHead: Selection
    ClusterHead --> GatewayNode: Inter-cluster Link
    MemberNode --> DataTransmission
    GatewayNode --> DataTransmission
    DataTransmission --> ChannelSensing: Periodic Update
```

### Q-Learning Process

```mermaid
sequenceDiagram
    participant SU as Secondary User
    participant ENV as Environment
    participant CH as Cluster Head
    
    SU->>ENV: Sense Channel
    ENV->>SU: Channel State
    SU->>SU: Calculate Reward
    SU->>SU: Update Q-Value
    SU->>CH: Share Q-Values
    CH->>CH: Evaluate Fitness
    CH->>SU: Cluster Decision
```

### Clustering Algorithm Flow

```mermaid
graph TD
    A[Start] --> B[Initialize Nodes]
    B --> C[Update Q-Values]
    C --> D[Calculate Fitness]
    D --> E{Is CH Candidate?}
    E -->|Yes| F[Become CH]
    E -->|No| G[Find Best CH]
    F --> H[Select CADC]
    G --> I[Join Cluster]
    H --> J[Assign Members]
    I --> K[Normal Operation]
    J --> K
```

Cognitive Radio Networks (CRN) allow secondary users (SUs) to opportunistically access licensed spectrum bands when primary users (PUs) are inactive. This implementation focuses on:

- Q-learning based channel quality evaluation
- Multi-objective cluster formation
- Dynamic spectrum allocation
- Energy-efficient network organization


## 🎯 Key Features

- **Q-Learning Based Channel Evaluation**: Dynamic channel quality assessment using reinforcement learning
- **Multi-Objective Clustering**: 
  - Residual energy optimization
  - Channel quality maximization
  - Network connectivity enhancement
- **Distributed Architecture**: No central control, fully autonomous nodes
- **Performance Metrics**: 
  - 30% improved network lifetime
  - 35% better energy efficiency
  - Enhanced spectrum utilization

## 🚀 Getting Started


1. Clone the repository:
```bash
git clone https://github.com/armanruet/Q-CRAHN.git
```

1. Install dependencies:
```bash
cd Q-CRAHN
pip install -r requirements.txt
```


### Quick Start

```python
from crahn.simulator import CRAHNSimulation

# Initialize simulation
sim = CRAHNSimulation(
    area_size=100,
    n_nodes=40,
    n_pus=12,
    n_channels=12
)

# Run simulation
metrics = sim.run_simulation(n_iterations=100)

# Visualize results
sim.visualize_network()
```

## 🗂️ Repository Structure

```
├── crahn/
│   ├── __init__.py
│   ├── simulator.py
│   ├── node.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{hossen2019qlearning,
  title={Q-Learning Based Multi-Objective Clustering Algorithm for Cognitive Radio Ad Hoc Networks},
  author={Hossen, Md Arman and Yoo, Sang-Jo},
  journal={IEEE Access},
  volume={7},
  pages={181959--181971},
  year={2019},
  publisher={IEEE}
}
```

## 🔬 Experimental Results

### Performance Analysis Architecture

```mermaid
graph LR
    subgraph "Metrics Collection"
        M1[Energy Efficiency]
        M2[Cluster Stability]
        M3[Spectrum Utilization]
    end
    
    subgraph "Analysis Tools"
        A1[Performance Monitor]
        A2[Network Analyzer]
        A3[Visualization Engine]
    end
    
    subgraph "Output"
        O1[Network Stats]
        O2[Performance Graphs]
        O3[Topology Maps]
    end
    
    M1 & M2 & M3 --> A1
    A1 --> A2
    A2 --> A3
    A3 --> O1 & O2 & O3
```

### Network Performance Metrics

```mermaid
gantt
    title Network Performance Timeline
    dateFormat X
    axisFormat %L
    
    section Energy
    CH Energy     :e1, 0, 100
    Member Energy :e2, 0, 80
    
    section Clustering
    Formation     :c1, 0, 20
    Stabilization :c2, 20, 60
    Operation     :c3, 60, 100
    
    section Spectrum
    Sensing      :s1, 0, 30
    Allocation   :s2, 30, 70
    Optimization :s3, 70, 100
```

### Cluster Formation
The simulation demonstrates effective cluster formation with the following characteristics:

- **Cluster Size**: Adaptive based on network conditions
- **Energy Distribution**: Balanced load across clusters
- **Spectrum Utilization**: Efficient channel assignment


## 🛠️ Implementation Details

### Q-Learning Parameters
- Learning Rate (α): 0.1
- Discount Factor (γ): 0.9
- ε-greedy Exploration: 0.1

### Network Parameters
- Area Size: 100m × 100m
- Number of Nodes: 40
- Number of Channels: 12
- Primary Users: 12
- Transmission Range: 30m

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any queries or suggestions, please reach out to:
- Email: armanruet@gmail.com
- LinkedIn: [armanruet](https://www.linkedin.com/in/armanruet/)

---
Made with ❤️ by [Arman](https://armanruet.github.io/)