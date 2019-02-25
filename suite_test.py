from dm_control import suite
import numpy as np
import cv2
import utils

env = suite.load(domain_name='cartpole', task_name='swingup')

env.reset()


saver = utils.VideoSaver("/home/duju/libtest.avi", 100, 640, 480)

for _ in range(1000):
    env.step(np.clip(np.random.randn(1),-1.0,1.0))

    frame = env.physics.render(camera_id=0, width=640, height=480)

    # FOR THE CASE OF RENDER DISPLAY
    cv2.imshow('Test', utils.RGB2BGR(frame))
    cv2.waitKey(delay=10)

    # FOR THE CASE OF SAVING VIDEO
    #saver.write(utils.RGB2BGR(frame))

saver.release()