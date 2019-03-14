import gym
import cv2
import utils

from gym.envs.mujoco import inverted_pendulum

env = inverted_pendulum.InvertedPendulumEnv()
print(env.dt)

#'InvertedPendulum-v2'
env = gym.make('Hopper-v2')
env.reset()


#saver = utils.VideoSaver("/home/duju/libtest.avi", source_fps=500, target_fps=30 , width=1000, height=1000)

print(env.dt)

for _ in range(10000):
    frame=env.render(mode='rgb_array',width=1000,height=1000)

    #FOR DISPLAY
    cv2.imshow('Test', utils.RGB2BGR(frame))
    cv2.waitKey(1)

    #FOR SAVE VIDEO
    #saver.write(utils.RGB2BGR(frame))

    env.step(env.action_space.sample())

#saver.release()