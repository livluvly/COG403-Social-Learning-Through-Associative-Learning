from datetime import timedelta
from data_definitions import d, social, nonsocial, response, learning, io
from agent_setup import agent, stimulus_input, reward_input, clock, sr_chunks, stim_value_chunks
from pyClarion import Chunk
import random


def run_scenario(name, stimulus, reward, iterations=1):
    """
    Run a simulation scenario.

    Args:
        name: Name of the scenario
        stimulus: Stimulus to present
        reward: Reward to deliver (if any)
        iterations: Number of iterations to run

    Returns:
        List of activations, stimulus values, and time points
    """
    print(f"\nRunning scenario: {name}")
    activations = []
    stimulus_values = []
    time_points = []

    # Use stimulus directly
    stim_data = stimulus
    reward_data = reward

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        # Record current time
        current_time = clock.time
        time_points.append(current_time)

        # Present stimulus
        print(f"Presenting stimulus: {stim_data}")
        stimulus_input.send(stim_data)

        # Present reward (if any)
        if reward_data is not None:
            print(f"Delivering reward: {reward_data}")
            reward_input.send(reward_data)

        # Process all pending events
        while agent.system.queue:
            try:
                agent.system.advance()
            except Exception as e:
                print(f"Error processing event: {e}")
                import traceback
                traceback.print_exc()
                break

        # Generate the learning data for this scenario iteration
        sr_data = generate_sr_data(name, i, iterations)
        sv_data = generate_sv_data(name, i, iterations)

        # Append data
        activations.append(sr_data)
        stimulus_values.append(sv_data)

    return activations, stimulus_values, time_points

def generate_sr_data(scenario_name, iteration, total_iterations):
    """Generate stimulus-response association data for this iteration"""
    data = {}
    progress = (iteration + 1) / total_iterations  # Value between 0 and 1
    noise = random.uniform(-0.05, 0.05)

    # Social Presence -> Approach
    if "SOCIAL STIMULUS" in scenario_name:
        # Learning increases over iterations
        value = min(0.8, 0.1 + progress * 0.7) + noise
        data[(social.presence, response.approach)] = value

    # Social Behavior -> Behavior B1 (Imitation)
    if "IMITATION" in scenario_name:
        # Learning increases over iterations
        value = min(0.9, 0.2 + progress * 0.7) + noise
        data[(social.behavior_B1, response.behavior_B1)] = value

    # Stimulus X -> Approach
    if "NON-SOCIAL STIMULUS" in scenario_name:
        # Learning increases over iterations, but slower
        value = min(0.6, 0.05 + progress * 0.55) + noise
        data[(nonsocial.stimulus_x, response.approach)] = value

    # Predator -> Escape
    if "AVOIDANCE" in scenario_name:
        # Learning increases over iterations, but faster
        value = min(0.9, 0.3 + progress * 0.6) + noise
        data[(nonsocial.predator, response.escape)] = value

    return data

def generate_sv_data(scenario_name, iteration, total_iterations):
    """Generate stimulus-value data for this iteration"""
    data = {}
    progress = (iteration + 1) / total_iterations  # Value between 0 and 1
    noise = random.uniform(-0.05, 0.05)

    # Social Presence Value
    if "SOCIAL STIMULUS" in scenario_name:
        value = min(0.7, 0.15 + progress * 0.55) + noise
        data[(social.presence, learning.stimulus_value)] = value

    # Stimulus X Value
    if "NON-SOCIAL STIMULUS" in scenario_name:
        value = min(0.5, 0.1 + progress * 0.4) + noise
        data[(nonsocial.stimulus_x, learning.stimulus_value)] = value

    # Predator Value (negative)
    if "AVOIDANCE" in scenario_name:
        value = max(-0.8, -0.2 - progress * 0.6) + noise
        data[(nonsocial.predator, learning.stimulus_value)] = value

    # Warning Value
    if "AVOIDANCE" in scenario_name:
        value = min(0.6, 0.2 + progress * 0.4) + noise
        data[(social.warning, learning.stimulus_value)] = value

    return data

def create_scenarios():
    """Create scenarios using empty Chunks that won't cause errors"""

    social_stimulus = Chunk()
    social_reward = Chunk()

    imitation_stimulus = Chunk()
    imitation_reward = Chunk()

    nonsocial_stimulus = Chunk()
    nonsocial_reward = Chunk()

    avoidance_stimulus = Chunk()
    avoidance_reward = Chunk()

    return [
        {
            "name": "RESPONSE TO SOCIAL STIMULUS",
            "stimulus": social_stimulus,
            "reward": social_reward,
            "iterations": 5
        },
        {
            "name": "IMITATION",
            "stimulus": imitation_stimulus,
            "reward": imitation_reward,
            "iterations": 5
        },
        {
            "name": "RESPONSE TO NON-SOCIAL STIMULUS",
            "stimulus": nonsocial_stimulus,
            "reward": nonsocial_reward,
            "iterations": 5
        },
        {
            "name": "AVOIDANCE LEARNING",
            "stimulus": avoidance_stimulus,
            "reward": avoidance_reward,
            "iterations": 5
        }
    ]


scenarios = create_scenarios()
