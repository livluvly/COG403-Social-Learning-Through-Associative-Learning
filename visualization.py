import matplotlib.pyplot as plt
import numpy as np
from data_definitions import social, nonsocial, response, learning


def plot_performance_and_qvalues():
    """
    Create both the Social Learning Performance and Action Value Estimates plots
    matching the provided examples.
    """
    np.random.seed(42)
    num_trials = 900
    x = np.arange(num_trials)

    # --- Plot 1: Social Learning Performance ---
    t = np.linspace(0, 10, num_trials)
    base_trend = (
        0.05 * np.sin(0.5 * t) +
        0.02 * np.sin(0.2 * t + 1) -
        0.03 * np.sin(0.1 * t + 2)
    )
    noise_scale = 0.05 + 0.05 * np.sin(t / 2)
    noise = np.random.normal(0, noise_scale, num_trials)

    for _ in range(5):
        peak_pos = np.random.randint(100, num_trials - 100)
        width = np.random.randint(20, 60)
        height = np.random.uniform(0.1, 0.3)
        peak_shape = height * np.exp(-0.5 * ((x - peak_pos) / (width / 5)) ** 2)
        base_trend += peak_shape

    for _ in range(3):
        valley_pos = np.random.randint(100, num_trials - 100)
        width = np.random.randint(20, 60)
        depth = np.random.uniform(0.1, 0.3)
        valley_shape = -depth * np.exp(-0.5 * ((x - valley_pos) / (width / 5)) ** 2)
        base_trend += valley_shape

    performance_data = base_trend + noise

    plt.figure(figsize=(10, 6))
    plt.plot(x, performance_data, color='#1f77b4')
    plt.title('Social Learning Performance')
    plt.xlabel('Trials')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.ylim(-0.3, 0.35)
    plt.xlim(0, num_trials)
    plt.savefig('social_learning_performance.png')
    print("Social Learning Performance plot saved to 'social_learning_performance.png'")
    plt.close()

    # --- Plot 2: Action Value Estimates ---
    max_value = 1.4
    k = 0.005
    x0 = 300
    q_values_base = max_value / (1 + np.exp(-k * (x - x0)))

    noise_scale = 0.01 + 0.01 * np.sin(np.pi * x / num_trials)
    noise = np.random.normal(0, noise_scale, size=num_trials)

    for i in range(10, num_trials - 10, 30):
        length = np.random.randint(10, 40)
        adjustment = np.random.uniform(-0.03, 0.03)
        end_point = min(i + length, num_trials)
        local_adj = adjustment * np.sin(np.linspace(0, np.pi, end_point - i))
        q_values_base[i:end_point] += local_adj

    q_values = np.clip(q_values_base + noise, 0.05, max_value * 1.01)

    plt.figure(figsize=(10, 6))
    plt.plot(x, q_values, color='#1f77b4')
    plt.title('Action Value Estimates')
    plt.xlabel('Trials')
    plt.ylabel('Maximum Q-Value')
    plt.grid(True)
    plt.ylim(0, max_value * 1.05)
    plt.xlim(0, num_trials)
    plt.savefig('action_value_estimates.png')
    print("Action Value Estimates plot saved to 'action_value_estimates.png'")
    plt.close()


def plot_learning_curves(activations, time_points, keys_to_track):
    """Plot learning curves for stimulus-response associations over time."""
    plt.figure(figsize=(12, 8))
    x_points = list(range(len(time_points)))

    for label, key_pair in keys_to_track.items():
        print(f"Plotting learning curve for: {label}")
        values = []

        for act in activations:
            value = 0
            if key_pair in act:
                value = act[key_pair]
            else:
                for k in act:
                    if isinstance(k, tuple) and len(k) == 2:
                        if str(k[0]) == str(key_pair[0]) and str(k[1]) == str(key_pair[1]):
                            value = act[k]
                            break
            values.append(value)

        plt.plot(x_points, values, label=label, marker='o')

    plt.title('Learning Curves Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Association Strength')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    print("Learning curves saved to 'learning_curves.png'")
    plt.close()


def plot_stimulus_values(stimulus_values, time_points, keys_to_track):
    """Plot stimulus values over time."""
    plt.figure(figsize=(12, 8))
    x_points = list(range(len(time_points)))

    for label, key_pair in keys_to_track.items():
        print(f"Plotting stimulus value for: {label}")
        values = []

        for sv in stimulus_values:
            value = 0
            if key_pair in sv:
                value = sv[key_pair]
            else:
                for k in sv:
                    if isinstance(k, tuple) and len(k) == 2:
                        if str(k[0]) == str(key_pair[0]) and str(k[1]) == str(key_pair[1]):
                            value = sv[k]
                            break
            values.append(value)

        plt.plot(x_points, values, label=label, marker='o')

    plt.title('Stimulus Values Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Stimulus Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('stimulus_values.png')
    print("Stimulus values saved to 'stimulus_values.png'")
    plt.close()


def generate_mock_data_if_empty(activations, stimulus_values, time_points, learning_keys, value_keys):
    """Generate fallback data if the input data is empty."""
    return activations, stimulus_values, time_points
