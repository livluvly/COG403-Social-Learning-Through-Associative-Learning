from pyClarion import Atom, Atoms, Family

# Data hierarchy for Social Learning through Associative Processes

class SocialStimuli(Atoms):
    """A sort for social stimuli terms."""
    presence: Atom  # Social presence
    behavior_B: Atom  # Social behavior
    behavior_B1: Atom  # Social behavior 1
    behavior_B2: Atom  # Social behavior 2
    warning: Atom  # Social warning signal

class NonSocialStimuli(Atoms):
    """A sort for non-social stimuli terms."""
    stimulus_x: Atom  # Non-social stimulus X
    stimulus_y: Atom  # Non-social stimulus Y
    predator: Atom  # Predator stimulus (threat)
    reward: Atom  # Reward stimulus

class Response(Atoms):
    """A sort for response terms."""
    behavior_B: Atom  # Generic behavior
    behavior_B1: Atom  # Specific behavior 1
    behavior_B2: Atom  # Specific behavior 2
    approach: Atom  # Approach response
    ignore: Atom  # Ignore/no response
    escape: Atom  # Escape/avoidance response

class Learning(Atoms):
    """A sort for learning-related terms."""
    stimulus_value: Atom  # Value associated with a stimulus
    response_value: Atom  # Value associated with a response
    reinforcement: Atom  # Reinforcement signal
    learning_rate_v: Atom  # Learning rate for values
    learning_rate_w: Atom  # Learning rate for weights

class IO(Atoms):
    """A sort for I/O related terms."""
    input: Atom  # Input channel
    output: Atom  # Output channel
    context: Atom  # Context information

class SocialLearningData(Family):
    """A family containing all data sorts for the social learning model."""
    social: SocialStimuli
    nonsocial: NonSocialStimuli
    response: Response
    learning: Learning
    io: IO

d = SocialLearningData()

social = d.social
nonsocial = d.nonsocial
response = d.response
learning = d.learning
io = d.io