from dm_control import suite
import cv2
import utils

env = suite.load(domain_name='cartpole', task_name='swingup')


for n_episode in range(100000):

    time_step = env.reset()

    # the input for actor network is 128,96 size single image
    observation = env.physics.render(camera_id=0, width=128, height=96)

    while not time_step.last():


        # the actor network outputs multi step actions
        action = actor.action(observation) + noise.noise()

        # proceed a step with action
        time_step = env.step(action)

        # Get reward
        reward = time_step.reward

        # Get next observation
        next_observation = env.physics.render(camera_id = 0, width = 128, height = 96)

        # Add transition tuple to replay buffer
        replay_buffer.add((observation,action, reward, next_observation))

        # Sample minibatch for training from replay buffer
        m_batch = replay_buffer.sample_minibatch()

        # Train network with minibatch
        DDPG.train(m_batch)

        # Move time_step
        observation = next_observation


class Actor:

class Noise:

class replay_buffer:

class DDPG: