from data_definitions import social, nonsocial, response, learning, io
from agent_setup import (
    agent, clock, stimulus_input, reward_input,
    sr_chunks, stim_value_chunks
)
from scenarios import scenarios, run_scenario
from visualization import (
    plot_learning_curves,
    plot_stimulus_values,
    plot_performance_and_qvalues
)


def main():
    """Run the social learning simulation."""
    print("Starting social learning simulation...")

    all_activations = []
    all_stimulus_values = []
    all_time_points = []

    # Run each scenario
    for scenario in scenarios:
        print(f"\nRunning scenario: {scenario['name']}")

        activations, stimulus_values, time_points = run_scenario(
            scenario["name"],
            scenario["stimulus"],
            scenario["reward"],
            scenario["iterations"]
        )

        all_activations.extend(activations)
        all_stimulus_values.extend(stimulus_values)
        all_time_points.extend(time_points)

    # Plot results
    print("\nGenerating plots...")
    plot_performance_and_qvalues()

    learning_keys = {
        'Social Presence -> Approach': (social.presence, response.approach),
        'Social Behavior -> Imitation': (social.behavior_B1, response.behavior_B1),
        'Stimulus X -> Approach': (nonsocial.stimulus_x, response.approach),
        'Predator -> Escape': (nonsocial.predator, response.escape)
    }

    value_keys = {
        'Social Presence Value': (social.presence, learning.stimulus_value),
        'Stimulus X Value': (nonsocial.stimulus_x, learning.stimulus_value),
        'Predator Value': (nonsocial.predator, learning.stimulus_value),
        'Warning Value': (social.warning, learning.stimulus_value)
    }

    plot_learning_curves(all_activations, all_time_points, learning_keys)
    plot_stimulus_values(all_stimulus_values, all_time_points, value_keys)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
