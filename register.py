from gymnasium.envs.registration import register

register(
    id="Env-energy-v2",  # Unique ID for your environment
    entry_point="env:MyEnv",  # Path to the environment class
)
