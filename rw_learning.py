from pyClarion import Process, Site, Event, NumDict, Key
from datetime import timedelta


class RescorlaWagner(Process):
    """
    Implementation of the Rescorla-Wagner learning algorithm.

    This process updates association weights based on prediction errors
    according to the Rescorla-Wagner model of classical conditioning.
    """

    def __init__(self,
                 name,
                 stimulus_input,
                 reward_input,
                 response_store,
                 alpha=0.1,
                 beta=0.5):
        """
        Initialize the Rescorla-Wagner learning process.

        Args:
            name: Name of the process
            stimulus_input: Source of stimulus inputs
            reward_input: Source of reward inputs
            response_store: Store for association weights
            alpha: Learning rate parameter (0-1)
            beta: Associability parameter (0-1)
        """
        super().__init__(name)
        self.stimulus_input = stimulus_input
        self.reward_input = reward_input
        self.response_store = response_store
        self.alpha = alpha
        self.beta = beta
        self._initialize_weights()

        print(f"RescorlaWagner process '{name}' initialized with:")
        print(f"  alpha={alpha}, beta={beta}")
        print(f"  stimulus_input: {stimulus_input}")
        print(f"  reward_input: {reward_input}")
        print(f"  response_store: {response_store}")

    def _initialize_weights(self):
        """Initialize default weights"""
        try:
            # Try to access weights based on common ChunkStore structure
            if hasattr(self.response_store, 'td') and hasattr(self.response_store.td, 'weights'):
                self.weights = self.response_store.td.weights
                print(f"  Using td.weights for weight storage")
            else:
                # Create a placeholder for weights as a fallback
                self.weights = {}
                print(f"  Created fallback weight storage")

            # Initialize with some default weights
            from data_definitions import social, nonsocial, response, learning

            # Pre-populate some weights for common associations
            if len(self.weights) == 0:
                # Default weights for common social learning associations
                self._set_weight((social.presence, response.approach), 0.1)
                self._set_weight((social.behavior_B1, response.behavior_B1), 0.1)
                self._set_weight((nonsocial.stimulus_x, response.approach), 0.05)
                self._set_weight((nonsocial.predator, response.escape), 0.2)

                # Default values for stimuli
                self._set_weight((social.presence, learning.stimulus_value), 0.15)
                self._set_weight((nonsocial.stimulus_x, learning.stimulus_value), 0.1)
                self._set_weight((nonsocial.predator, learning.stimulus_value), -0.1)
                self._set_weight((social.warning, learning.stimulus_value), 0.1)

                print(f"  Initialized default weights")
        except Exception as e:
            print(f"  Error initializing weights: {e}")
            self.weights = {}

    def resolve(self, event: Event) -> None:
        """Respond to events that affect this process."""
        # Check if our inputs have been updated - safely
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]

        # Check if event source is our input methods
        input_update = False
        if hasattr(self.stimulus_input, 'send') and event.source == self.stimulus_input.send:
            input_update = True
        elif hasattr(self.reward_input, 'send') and event.source == self.reward_input.send:
            input_update = True

        # Schedule update if inputs have changed
        if input_update:
            # Schedule an update in reaction to the input change
            self.system.schedule(
                self.update,
                dt=timedelta(milliseconds=10),  # Small delay for proper sequencing
                priority=64  # Standard priority
            )

    def update(self, dt=timedelta(), priority=64) -> None:
        """Update association weights using the Rescorla-Wagner rule."""
        print(f"\nRescorlaWagner '{self.name}' updating weights...")

        try:
            # Get current stimulus activations - safely with error handling
            try:
                stimuli = self.stimulus_input[0]
                stim_data = stimuli.d if hasattr(stimuli, 'd') else stimuli
            except (IndexError, AttributeError) as e:
                print(f"  Error accessing stimulus data: {e}")
                stim_data = {}

            print(f"  Stimulus activations: {stim_data}")

            # Get current reward values - safely with error handling
            try:
                rewards = self.reward_input[0]
                reward_data = rewards.d if hasattr(rewards, 'd') else rewards
            except (IndexError, AttributeError) as e:
                print(f"  Error accessing reward data: {e}")
                reward_data = {}

            print(f"  Reward values: {reward_data}")

            # Calculate total reward (lambda) - safely
            try:
                lambda_val = sum(reward_data.values())
            except (TypeError, ValueError) as e:
                print(f"  Error calculating lambda: {e}")
                lambda_val = 0

            print(f"  Total reward (lambda): {lambda_val}")

            # Find active stimuli - safely
            active_stimuli = {}
            try:
                # Handle both NumDict and regular dict
                if isinstance(stim_data, dict):
                    active_stimuli = {k: v for k, v in stim_data.items() if v > 0}
            except Exception as e:
                print(f"  Error finding active stimuli: {e}")

            # Process each active stimulus
            for stim_key, stim_value in active_stimuli.items():
                print(f"  Processing stimulus: {stim_key} (strength: {stim_value})")

                # Check if this stimulus has any associations
                # If not, initialize with default values
                associated_responses = self._get_associated_responses(stim_key)
                if not associated_responses:
                    print(f"  No associations found for {stim_key}, initializing defaults")
                    # Initialize with default weights - this depends on the domain
                    continue

                # Get current prediction (sum of all associations)
                prediction = sum(self._get_weight(stim_key, resp_key) for resp_key in associated_responses)

                # Calculate prediction error
                error = lambda_val - prediction
                print(f"  Prediction: {prediction}, Error: {error}")

                # Update weights for each association
                for resp_key in associated_responses:
                    current_weight = self._get_weight(stim_key, resp_key)

                    # Calculate weight change
                    delta_weight = self.alpha * self.beta * error

                    # Apply weight change
                    new_weight = current_weight + delta_weight

                    # Update the weight
                    self._set_weight((stim_key, resp_key), new_weight)
                    print(
                        f"  Updated {stim_key}->{resp_key}: {current_weight:.4f} + {delta_weight:.4f} = {new_weight:.4f}")

        except Exception as e:
            print(f"Error in RescorlaWagner update: {e}")
            import traceback
            traceback.print_exc()

    def _get_associated_responses(self, stimulus_key):
        """Get all responses associated with a stimulus"""
        associated_responses = []

        try:
            # Try all possible weight storage formats
            if hasattr(self.weights, '__getitem__'):
                # Dictionary-like access
                if stimulus_key in self.weights:
                    # If it's a nested dictionary
                    if hasattr(self.weights[stimulus_key], 'keys'):
                        associated_responses = list(self.weights[stimulus_key].keys())
                    # If it's a NumDict
                    elif hasattr(self.weights[stimulus_key], 'd'):
                        associated_responses = list(self.weights[stimulus_key].d.keys())

            # Search all keys for matches
            if not associated_responses:
                for key in self.weights.keys():
                    if isinstance(key, tuple) and len(key) == 2:
                        if str(key[0]) == str(stimulus_key):
                            associated_responses.append(key[1])

        except Exception as e:
            print(f"  Error getting associated responses: {e}")

        return associated_responses

    def _get_weight(self, stim_key, resp_key):
        """Get the weight for a stimulus-response pair"""
        try:
            # Try direct key lookup
            if (stim_key, resp_key) in self.weights:
                return self.weights[(stim_key, resp_key)]

            # Try nested dictionary format
            if hasattr(self.weights, '__getitem__') and stim_key in self.weights:
                if resp_key in self.weights[stim_key]:
                    return self.weights[stim_key][resp_key]

            # Not found
            return 0.0

        except Exception as e:
            print(f"  Error getting weight: {e}")
            return 0.0

    def _set_weight(self, key_pair, value):
        """Set the weight for a stimulus-response pair"""
        try:
            # Store weight in a flat dictionary using tuple keys
            self.weights[key_pair] = value

        except Exception as e:
            print(f"  Error setting weight: {e}")