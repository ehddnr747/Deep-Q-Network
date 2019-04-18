import numpy as np

#Gaussian Noise for action noise
class GaussianNoise:
    def __init__(self,action_dim, sigma):
        self.action_dim = action_dim
        self.sigma = sigma

    def __call__(self):
        return np.random.randn(self.action_dim) * self.sigma

    def reset(self):
        pass

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None, actions_per_control = 1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
        self.actions_per_control = actions_per_control

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def __call__(self):
        if self.actions_per_control == 1: # return 1d array
            return self.sample()
        else: # return 2d array
            noises = []
            for _ in range(self.actions_per_control):
                 noises.append(self.sample())
            return np.array(noises)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)