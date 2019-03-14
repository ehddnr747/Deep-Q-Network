import tflearn
import tensorflow as tf
import ReplayBuffer
import numpy as np
import Actor
import Noise
import Critic
from dm_control import suite
import cv2
import utils
import os



def train_feature(sess, env, actor, critic, actor_noise, batch_size,saver, video_dir,step_size):

    #training with low dimensional features

    sess.run(tf.global_variables_initializer())

    actor.initialize_target()
    critic.initialize_target()

    replay_buffer = ReplayBuffer.ReplayBuffer(1000000)

    video_save_period =50


    for i in range(1000):

        #if i%1000 == 0:
        #    saver.save(sess,"/home/duju/git_repos/model.ckpt")

        time_step = env.reset()
        _, _, _, s = time_step
        s = utils.state_1d_flat(s)
        # in the case of suite [step_type, reward, discount, observation]
        # in the case of gym [observation, reward, done, info], gym reset returns observation only

        ep_reward = 0

        if i%video_save_period == 0:
            video_saver = utils.VideoSaver(os.path.join(video_dir,"training_"+str(i)+".avi")\
                                           , int(1. / env.control_timestep()), 30, width=320, height=240)

        while time_step.last() != True:

            a = actor.predict(np.reshape(s, (1, *actor.state_dim)))
            a[0] = np.clip(a[0] + actor_noise(),-1.0,1.0)
            # a : [?, action_dim]


                a = actor.predict(np.reshape(s, (1, *actor.state_dim)))


            for _ in range(step_size):
                time_step = env.step(a[0])

                if (i % video_save_period == 0):
                    frame = env.physics.render(camera_id=0, width=320, height=240)
                    video_saver.write(utils.RGB2BGR(frame))

            terminal, r, _, s2 = time_step
            s2 = utils.state_1d_flat(s2)
            r = 0.1 * r

            replay_buffer.add(s,np.reshape(a, actor.action_dim),r, terminal, s2)
            # s : [4], a : [1], r: scalar, done : scalar, s2 : [4]

            if replay_buffer.size() > batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(batch_size)

                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch)
                )
                # taget_q : [batch_size, 1]
                y_i = []


                for k in range(batch_size):

                    #no consideration for terminal
                    y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # y_i : [batch_size, 1]

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
                print(i,"---",ep_reward)
                utils.reward_writer(video_dir,i,ep_reward)

        if i % video_save_period == 0:
            video_saver.release()

if __name__ == '__main__':

    batch_size = 100

    tf_config = tf.ConfigProto()

    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    tf_config.gpu_options.allow_growth = True

    domain_name = "cartpole"
    task_name = "swingup"
    step_size = 10

    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={"time_limit":10*step_size})
    #step outputs [termianl, reward, discount, obesrvation]

    video_dir = utils.directory_setting("/home/duju/training",domain_name,task_name)

    saver = tf.train.Saver()
    state_dim = utils.state_1d_dim_calc(env)

    actor_lr = 1e-3
    critic_lr = 1e-3
    tau = 5e-3
    gamma = 0.99

    with tf.Session(config=tf_config) as sess:
                        #state_dim : 1d, action_spec : scalar
        actor = Actor.Actor(sess, state_dim, env.action_spec().shape[0], actor_lr, tau, batch_size)
        critic = Critic.Critic(sess, state_dim, env.action_spec().shape[0], critic_lr, tau, gamma, actor.get_num_trainable_vars())

        actor_noise = Noise.GaussianNoise(action_dim=env.action_spec().shape[0],sigma=0.1)

        train_feature(sess, env, actor, critic, actor_noise, batch_size,saver, video_dir,step_size)