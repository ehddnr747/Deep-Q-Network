import tensorflow as tf
import ReplayBuffer
import numpy as np
import Actor
import Noise
import Critic
import gym
import cv2
import utils
import time


"""
def train_vision(sess, env, actor, critic, actor_noise, batch_size,saver):

    #image size :

    sess.run(tf.global_variables_initializer())

    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer.ReplayBuffer(100000)


    for i in range(100000):

        if i%100 == 0:
            saver.save(sess,"/home/duju/git_repos/model.ckpt")

        time_step = env.reset()


        ep_reward = 0

        s = env.physics.render(camera_id=0, width=32, height=24)

        while True:

            cv2.imshow('Test', utils.RGB2BGR(s))
            cv2.waitKey(delay=1)

            a = np.reshape(actor.predict(np.reshape(s, (1, *actor.state_dim))) + actor_noise(), actor.action_dim)


            time_step = env.step(a[0])
            terminal, r, _, _ = time_step


            s2 = env.physics.render(camera_id=0, width=32, height=24)

            replay_buffer.add(s,a,r, terminal, s2)

            if replay_buffer.size() > batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(batch_size)

                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch)
                )

                y_i = []

                for k in range(batch_size):

                    if t_batch[k].last():
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])



                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i,(batch_size,1))
                )

                a_outs = actor.predict(s_batch)
                
                grads = critic.action_gradients(s_batch,a_outs)
                actor.train(s_batch, grads[0]) #grads is returned as list of length 1

                actor.update_target_network() # Do we do this every time?
                critic.update_target_network()

                s = s2

                ep_reward += r

            if time_step.last():
                print(ep_reward)
                break
"""
def train_feature(sess, env, actor, critic, actor_noise, batch_size,saver):

    #training with low dimensional features

    sess.run(tf.global_variables_initializer())

    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer.ReplayBuffer(1000000)


    for i in range(1000000000):

        #if i%1000 == 0:
        #    saver.save(sess,"/home/duju/git_repos/model.ckpt")

        s = env.reset()
        # in the case of suite [step_type, reward, discount, observation]
        # in the case of gym [observation, reward, done, info], gym reset returns observation only

        ep_reward = 0

        #if i%20 == 0:
        #    pass
            #video_saver = utils.VideoSaver("/home/duju/git_repos/training.avi", int(1. / env.control_timestep()), 30, width=320, height=240)


        done = False

        while done != True:

            a = actor.predict(np.reshape(s, (1, *actor.state_dim))) + actor_noise()
            a = 3 * a

            if(i%100 == 0):
            #    pass
                #env.render()
                a = actor.predict(np.reshape(s, (1, *actor.state_dim)))
                a = 3 * a
                #frame = env.physics.render(camera_id=0, width=320, height=240)
                #video_saver.write(utils.RGB2BGR(frame))

            #frame = env.physics.render(camera_id=0, width=320, height=240)
            #cv2.imshow('Test', utils.RGB2BGR(frame))
            #cv2.waitKey(delay=1)


            env.render()

            # a : [?, action_dim]


            time_step = env.step(a[0])
            s2, r, done, _ = time_step

            replay_buffer.add(s,np.reshape(a, actor.action_dim),r, done, s2)
            #print((s,np.reshape(a, actor.action_dim),r, done, s2))
            # s : [4], a : [1], r: scalar, done : scalar, s2 : [4]



            if replay_buffer.size() > batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)

                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch)
                )
                # taget_q : [batch_size, 1]
                y_i = []


                for k in range(batch_size):

                    if t_batch[k]:
                        y_i.append(np.reshape(r_batch[k],(1,)))
                        #print("ya", r_batch[k])
                    else:
                        #print(r_batch[k], critic.gamma, target_q[k])
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                    #print(a_batch[k], r_batch[k],y_i[k])

                # y_i : [?, batch_size]

                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i,(batch_size,1))
                )
                #print(predicted_q_value[0],y_i[0])

                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch,a_outs)
                #print(s_batch.shape)
                #print(a_outs.shape)
                #print(sess.run(critic.network_params[5],feed_dict={critic.inputs: s_batch, critic.action: a_outs})[:10])
                #print(sess.run(tf.gradients(critic.out,critic.inputs),feed_dict={critic.inputs: s_batch,
                #critic.action: a_outs})
                #)
                #print(np.array(grads).shape)


                actor.train(s_batch, grads[0]) #grads is returned as list of length 1

                actor.update_target_network() # Do we do this every time?
                critic.update_target_network()

                s = s2

                ep_reward += r

            if done:
                print(i,"---",ep_reward)
                break

        if i % 20 == 0:
            pass
            #video_saver.release()


if __name__ == '__main__':

    batch_size = 64
    tf_config = tf.ConfigProto()
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    tf_config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    env = gym.make('InvertedPendulum-v2')

    #print(env.action_space.high)

    with tf.Session(config=tf_config) as sess:

        actor = Actor.Actor(sess, [4], 1, 0.0001 , 0.001, batch_size)
        critic = Critic.Critic(sess, [4], 1, 0.001, 0.001, 0.99, actor.get_num_trainable_vars())
        actor_noise = Noise.GaussianNoise()

        train_feature(sess, env, actor, critic, actor_noise, batch_size,saver)