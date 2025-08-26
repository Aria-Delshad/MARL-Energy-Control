# MARL-Energy-Control
# Simulation of of Multi-Agent Control for Shared Energy Storage

This project provides a MATLAB implementation of the core concepts from the paper "Learning a Multi-Agent Controller for Shared Energy Storage System" by Ruoheng Liu and Yize Chen. The simulation is built from scratch and **does not require any MATLAB toolboxes**.

## Description

This project benchmarks four different control strategies for a community microgrid consisting of two buildings and a shared energy storage system (SESS):
1.  **Heuristic Baseline:** A simple, rule-based controller.
2.  **User-Only Baseline:** Buildings operate independently without the SESS.
3.  **Centralized DDPG Agent:** A single Deep Deterministic Policy Gradient (DDPG) agent that controls all components.
4.  **Proposed MADDPG Agent:** A Multi-Agent DDPG controller where each building and the SESS is a separate agent.



## How to Run

1.  Ensure all `.m` files are in the same directory.
2.  Open MATLAB and navigate to this directory.
3.  Run the main script from the MATLAB Command Window:
    ```matlab
    run_advanced_simulation
    ```
4.  The script will train the RL agents and evaluate all four methods. Progress will be printed to the console.
5.  Upon completion, all results, including a summary table (`results_summary.csv`) and plots, will be saved in the `outputs_advanced` folder.

**Note:** The default number of training episodes is set to 50 for a quick demonstration. For better convergence and performance, this can be increased inside the `run_advanced_simulation.m` script.

## Project Structure

This project is organized into modular class-based files:

-   `run_advanced_simulation.m`: The main executable script that orchestrates the entire simulation and benchmarking process.
-   `CommunityEnv.m`: A class defining the simulation environment, including the building thermal dynamics, the SESS model, and state/reward calculations.
-   `FeedForwardNetwork.m`: A custom class for a simple feed-forward neural network, complete with forward/backward propagation and an Adam optimizer.
-   `ReplayBuffer.m`: A class for the experience replay buffer used by the RL agents.
-   `CentralizedDDGPAgent.m`: A class implementing the single-agent DDPG algorithm for the centralized baseline.
-   `MADDPGAgent.m`: A class implementing the Multi-Agent DDPG algorithm as described in the paper.
-   `README.md`: This file.
