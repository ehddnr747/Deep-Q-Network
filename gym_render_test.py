import gym
import time

env = gym.make('Hopper-v2')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
    time.sleep(0.005)
