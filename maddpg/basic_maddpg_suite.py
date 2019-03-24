import sys
sys.path.append("../models")
sys.path.append("../utils")

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
import graph_reward
import RNNActor




def train_basic_maddpg(num_of_action,sess, env, actor, critic, actor_noise, batch_size,saver, video_dir,step_size):

    #training with low dimensional features

    sess.run(tf.global_variables_initializer())

    actor.initialize_target()
    critic.initialize_target()

    replay_buffer = ReplayBuffer.ReplayBuffer(1000000)

    video_save_period = 10

    single_action_dim = int(actor.action_dim / num_of_action)


    for i in range(1,1000+1):

        #if i%1000 == 0:
        #    saver.save(sess,"/home/duju/git_repos/model.ckpt")

        time_step = env.reset()
        _, _, _, s = time_step
        s = utils.state_1d_flat(s)
        # in the case of suite [step_type, reward, discount, observation]
        # in the case of gym [observation, reward, done, info], gym reset returns observation only

        ep_reward = 0

        actor_noise.reset()

        if i%video_save_period == 0:
            video_saver = utils.VideoSaver(os.path.join(video_dir,"training_"+str(i)+".avi")\
                                           , int(1. / env.control_timestep())*step_size, 30, width=320, height=240)

        while time_step.last() != True:

            a = actor.predict(np.reshape(s, (1, *actor.state_dim)))

            a = np.reshape(a,(num_of_action,single_action_dim))

            if (i % video_save_period != 0):
                a_noise = np.reshape(actor_noise(),(1,single_action_dim))
                a = np.clip(a + a_noise,-1.0,1.0)
            # a : [?, action_dim]
            if (i <= 10):
                a_noise = np.reshape(actor_noise(),(1,single_action_dim))
                a = np.clip(np.zeros([num_of_action,single_action_dim])+a_noise,-1.0,1.0)

            for a_i in range(num_of_action):
                time_step = env.step(a[a_i])

                if (i % video_save_period == 0):
                    frame = env.physics.render(camera_id=0, width=320, height=240)
                    video_saver.write(utils.RGB2BGR(frame))

                #break multistep action
                if time_step.last():
                    break
            #break episode
            if time_step.last():
                break

            terminal, r, _, s2 = time_step
            s2 = utils.state_1d_flat(s2)

            replay_buffer.add(s,np.reshape(a, actor.action_dim),r, terminal, s2)
            # s : [4], a : [1], r: scalar, done : scalar, s2 : [4]

            for _ in range(num_of_action):
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

                    actor.update_target_network()
                    critic.update_target_network()

            s = s2
            ep_reward += r


        max_q_from_laststep = np.max(predicted_q_value)
        print(i,"***",ep_reward,"***",max_q_from_laststep)
        utils.line_writer(video_dir,\
                        str(i)+" *** "+str(ep_reward)+" *** "+str(max_q_from_laststep)+"\n")

        if i % video_save_period == 0:
            video_saver.release()
            graph_reward.save_graph(os.path.basename(video_dir),reward_scale = int(1000/num_of_action))


if __name__ == '__main__':

    batch_size = 100

    tf_config = tf.ConfigProto()

    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    tf_config.gpu_options.allow_growth = True

    domain_name = "cartpole"
    task_name = "swingup_sparse"

    env_temp = suite.load(domain_name=domain_name,task_name=task_name)
    control_timestep = env_temp.control_timestep()
    del env_temp

    step_size = 1
    time_limit = 1000*control_timestep*step_size

    env = suite.load(domain_name=domain_name, task_name=task_name, \
                     task_kwargs={"time_limit":time_limit})
    #step outputs [termianl, reward, discount, obesrvation]

    video_dir = utils.directory_setting("/home/duju/training",domain_name,task_name,step_size)

    saver = tf.train.Saver()
    state_dim = utils.state_1d_dim_calc(env)

    num_of_action = 2

    actor_lr = 1e-5
    critic_lr = 1e-5
    tau = 1e-3
    gamma = 0.99
    sigma = 0.2
    critic_reg_weight = 0.0
    noise_type = "ou"
    action_dim = env.action_spec().shape[0]*num_of_action
    actor_type = "rnn"

    assert noise_type in ["ou","gaussian"]
    assert actor_type in ["rnn","basic"]

    with tf.Session(config=tf_config) as sess:
                        #state_dim : 1d, action_spec : scalar

        if actor_type == "basic":
            actor = Actor.Actor(sess, state_dim, action_dim, actor_lr, tau, batch_size)
        elif actor_type == "rnn":
            actor = RNNActor.Actor(sess, state_dim, action_dim, actor_lr, tau, batch_size, num_of_action)

        critic = Critic.Critic(sess, state_dim, action_dim, critic_lr, tau, gamma, actor.get_num_trainable_vars(),critic_reg_weight)

        if noise_type == "gaussian":
            actor_noise = Noise.GaussianNoise(action_dim=action_dim,sigma=sigma)
        elif noise_type == "ou":
            actor_noise = Noise.OrnsteinUhlenbeckActionNoise(mu=np.zeros([int(action_dim/num_of_action)]), sigma=sigma)

        exp_detail = utils.experiment_detail_saver(
                            domain_name, task_name, step_size,
                            actor_lr, critic_lr, tau,
                            gamma, sigma, batch_size,
                            critic_reg_weight)

        print(exp_detail)
        utils.append_file_writer(video_dir, "experiment_detail.txt", "num of action : " \
                        + str(num_of_action) + "\n")
        print("num of action : " + str(num_of_action))

        utils.append_file_writer(video_dir, "experiment_detail.txt", "actor type : " \
                         + actor_type + "\n")
        print("actor type : " + actor_type +"\n")

        utils.append_file_writer(video_dir, "experiment_detail.txt", "Critic origin type : "\
                                 +critic.critic_origin_type+"\n")
        utils.append_file_writer(video_dir, "experiment_detail.txt", "Noise type : " \
                                 + noise_type + "\n")

        utils.append_file_writer(video_dir, "experiment_detail.txt",exp_detail)


        train_basic_maddpg(num_of_action, sess, env, actor, critic, actor_noise, batch_size,saver, video_dir,step_size)

