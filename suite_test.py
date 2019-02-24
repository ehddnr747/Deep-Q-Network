from dm_control import suite
import numpy as np
import cv2

env = suite.load(domain_name='cartpole', task_name='swingup')

env.reset()


jfourcc = cv2.VideoWriter_fourcc(*'XVID')
jout = cv2.VideoWriter("/home/duju/test.avi", jfourcc, 100, (640,480))


for _ in range(1000):
    env.step(np.clip(np.random.randn(1),-1.0,1.0))

    # FOR THE CASE OF RENDER DISPLAY
    #cv2.imshow('Test', cv2.cvtColor(env.physics.render(camera_id=0),cv2.COLOR_RGB2BGR))
    #cv2.waitKey(delay=10)

    # FOR THE CASE OF SAVING VIDEO
    jout.write(cv2.cvtColor(env.physics.render(height=480 ,width=640, camera_id=0),cv2.COLOR_RGB2BGR))

jout.release()