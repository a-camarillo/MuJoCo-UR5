from gymnasium.envs.registration import register

register(
        id='manipulator_env/Manipulator-v0',
        entry_point="manipulator_env.envs:ManipulatorEnv"
)
