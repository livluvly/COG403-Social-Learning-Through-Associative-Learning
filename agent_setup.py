from pyClarion import Agent, Input, ChunkStore, Clock
from data_definitions import d, social, nonsocial, response, learning, io
from rw_learning import RescorlaWagner

clock = Clock()
agent = Agent("social_learning_agent", d=d)

with agent:
    stimulus_input = Input("stimulus_input", (d.io, d))
    reward_input = Input("reward_input", (d.learning, d))

    # Stimulus-response associations
    sr_chunks = ChunkStore(
        "sr_chunks",
        c=d,
        d=d.io,
        v=d.response
    )

    # Stimulus-value associations
    stim_value_chunks = ChunkStore(
        "stim_value_chunks",
        c=d,
        d=d,
        v=d.learning
    )

    # Rescorla-Wagner learning for stimulus-response associations
    rw_sr = RescorlaWagner(
        name="rw_sr_learner",
        stimulus_input=stimulus_input,
        reward_input=reward_input,
        response_store=sr_chunks,
        alpha=0.2,
        beta=0.5
    )

    # Rescorla-Wagner learning for stimulus values
    rw_val = RescorlaWagner(
        name="rw_val_learner",
        stimulus_input=stimulus_input,
        reward_input=reward_input,
        response_store=stim_value_chunks,
        alpha=0.15,
        beta=0.5
    )

print("Agent initialized with the following components:")
print(f"- stimulus_input: {stimulus_input}")
print(f"- reward_input: {reward_input}")
print(f"- sr_chunks: {sr_chunks}")
print(f"- stim_value_chunks: {stim_value_chunks}")
print(f"- rw_sr: {rw_sr}")
print(f"- rw_val: {rw_val}")
