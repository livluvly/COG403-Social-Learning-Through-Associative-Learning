# Social Learning Through Associative Learning Simulation

This repository contains a simulation of social learning behaviors modeled through associative learning mechanisms, using the Rescorla-Wagner algorithm within the pyClarion cognitive architecture. It demonstrates how agents learn to respond to both social and non-social cues using reinforcement-driven processes.

## Repository Structure

- `main.py`: Orchestrates the full simulation and plotting pipeline.
- `agent_setup.py`: Initializes the agent, inputs, and learning modules.
- `data_definitions.py`: Defines the hierarchical structure for stimuli, responses, and learning terms.
- `rw_learning.py`: Implements the Rescorla-Wagner learning process.
- `scenarios.py`: Specifies and executes learning scenarios.
- `visualization.py`: Generates visual outputs showing performance and learning trends.
- `README.md`: This file.

## Overview

The simulation models four key aspects of social learning:

1. Response to social stimuli
2. Imitation learning
3. Response to non-social stimuli
4. Avoidance learning through social warning signals

The model integrates two types of associative learning:
- **Stimulus-response learning**
- **Stimulus-value learning**

These are implemented via two Rescorla-Wagner processes.

## Running the Simulation

Ensure dependencies are installed:
```bash
pip install pyClarion numpy matplotlib
```

Run the simulation:
```bash
python main.py
```

## Output

The simulation generates several plots saved as PNG files:

1. `action_value_estimates.png`: Tracks the agent's action value estimates across trials.
2. `social_learning_performance.png`: Displays average reward trends over time.
3. `stimulus_values.png`: Visualizes stimulus values as they evolve.
4. `learning_curves.png`: Tracks learned stimulus-response associations over time.

## Model Parameters

The Rescorla-Wagner learners are configured as follows:

- **Alpha (α)**: Learning rate
  - Stimulus-response learner: α = 0.2
  - Stimulus-value learner: α = 0.15

- **Beta (β)**: Associability
  - Both learners: β = 0.5

## Learning Scenarios

1. **Response to Social Stimulus**
   - Models learning to approach social presence based on positive reinforcement.

2. **Imitation Learning**
   - Models reinforcement-based imitation of observed social behaviors.

3. **Response to Non-Social Stimulus**
   - Evaluates associative learning in the presence of environmental cues.

4. **Avoidance Learning**
   - Trains the agent to avoid threats using social warning signals.

## Key Findings

- Social stimuli enable faster learning than non-social ones.
- Imitation supports efficient acquisition of adaptive responses.
- Social cues like warnings can facilitate learning without direct exposure.
- The Rescorla-Wagner framework successfully models diverse social learning mechanisms.

## Dependencies

- `pyClarion`
- `numpy`
- `matplotlib`