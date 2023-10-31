import gym

# Initialize the environment
env = gym.make('CartPole-v1')

# Reset the environment to start
observation = env.reset()

# Run for 100 time-steps
for t in range(100):
    # Render the environment to visualize it
    env.render()

    # Take a random action from the action space
    action = env.action_space.sample()

    # Apply the action and get the new state and reward
    step_result = env.step(action)
    observation, reward, done, info = step_result[:4]

    # If the episode is done, reset the environment
    if done:
        print(f"Episode finished after {t+1} timesteps")
        break

# Close the environment
env.close()
