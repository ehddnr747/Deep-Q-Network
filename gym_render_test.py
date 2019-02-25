import gym
import cv2
import utils

#'InvertedPendulum-v2'
env = gym.make('Hopper-v2')
env.reset()


saver = utils.VideoSaver("/home/duju/libtest.avi", 500, 1000, 1000)

for _ in range(10000):
    frame=env.render(mode='rgb_array',width=1000,height=1000)

    #FOR DISPLAY
    #cv2.imshow('Test', utils.RGB2BGR(frame))
    #cv2.waitKey(1)

    #FOR SAVE VIDEO
    saver.write(utils.RGB2BGR(frame))

    env.step(env.action_space.sample())

saver.release()