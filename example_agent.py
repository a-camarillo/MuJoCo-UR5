import gymnasium
import manipulator_env

env = gymnasium.make('manipulator_env:manipulator_env/Manipulator-v0', render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
