import tflearn
import tensorflow as tf

class Critic(object):
    """
    Input (s,a)
    Output Q(s,a)
    """

    def __init__(self,sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess= sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params)+num_actor_vars):]


        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)\
                                                  + tf.multiply(self.target_network_params[i],1. - self.tau))
                for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.out, self.predicted_q_value)

        self.optimize = tf.train.AdadeltaOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)


    def create_critic_network(self):

        if len(self.state_dim) == 3:
            inputs = tflearn.input_data(shape=[None, *self.state_dim])
            action = tflearn.input_data(shape=[None, self.action_dim])

            net = tflearn.conv_2d(incoming=inputs, nb_filter=32, filter_size=7, activation='ReLU')
            net = tflearn.conv_2d(incoming=net, nb_filter=32, filter_size=7, activation='ReLU')
            net = tflearn.conv_2d(incoming=net, nb_filter=32, filter_size=7, activation='ReLU')

            t1 = tflearn.fully_connected(net, 300, activation='ReLU')
            t2 = tflearn.fully_connected(action, 300, activation='ReLU')

            net = tflearn.layers.merge_ops.merge([t1,t2],mode="concat",axis=1)

            w_init = tflearn.initializations.uniform(minval=-0.003,maxval=0.003)

            out = tflearn.fully_connected(net, 1, weights_init=w_init)

            return inputs, action, out


        else:
            assert 0 == 1, "error in create_critic_network, state_dim not matches"

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs : inputs,
            self.action : action,
            self.predicted_q_values: predicted_q_value
        })


    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self,inputs,action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.taget_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
