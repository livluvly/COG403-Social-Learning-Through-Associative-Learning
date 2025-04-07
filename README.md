# Social Learning Through Associative Learning Simulation

This project implements a computational model of social learning through associative processes, based on the Rescorla-Wagner learning algorithm. The simulation demonstrates how social and non-social stimuli can influence learning and behavior through different associative pathways.

## Overview

The simulation models 4 key aspects of social learning:

1. Response to social stimuli
2. Imitation learning
3. Response to non-social stimuli
4. Avoidance learning through social warning signals

Using the pyClarion cognitive architecture framework, the model incorporates two Rescorla-Wagner learning processes:
- Stimulus-response associations
- Stimulus-value associations

## Components

The simulation consists of several key components:

- **Agent Setup**: Initializes the agent with the learning architecture
- **Data Definitions**: Defines the social and non-social stimuli, responses, and learning parameters
- **Rescorla-Wagner Learning**: Implements the associative learning algorithm
- **Scenarios**: Defines and runs different learning scenarios
- **Visualization**: Generates plots showing learning performance and outcomes

## Running the Simulation

To run the simulation:

```bash
python main.py
```

## Output

The simulation produces several visualizations:

1. **Action Value Estimates**: Shows how action values increase over time as the agent learns
2. **Social Learning Performance**: Shows the pattern of rewards received during learning
3. **Stimulus Values Over Time**: Shows how different stimuli acquire different values through learning

## Model Parameters

The Rescorla-Wagner learning process has two primary parameters:

- **Alpha (α)**: Learning rate - determines how quickly associations are formed
  - sr_learner: α = 0.2
  - val_learner: α = 0.15

- **Beta (β)**: Associability - determines the influence of prediction errors
  - Both learners: β = 0.5

## Learning Scenarios

### 1. Response to Social Stimulus
Models how the agent learns to respond to social presence with approach behavior.

### 2. Imitation Learning
Models how the agent learns to imitate observed social behaviors.

### 3. Response to Non-Social Stimulus
Models learning responses to non-social stimuli for comparison.

### 4. Avoidance Learning
Models how social warning signals help the agent learn to avoid predator threats.

## Analysis

The simulation demonstrates several key principles of social learning:

1. Social stimuli can facilitate faster learning compared to non-social stimuli
2. Imitation provides an efficient mechanism for acquiring adaptive behaviors
3. Social warning signals can facilitate avoidance learning without direct experience
4. The Rescorla-Wagner algorithm effectively captures the associative processes underlying these learning mechanisms

## Dependencies

- pyClarion
- NumPy
- Matplotlib
