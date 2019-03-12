import numpy as np

class GaussianNoise:
    def __init__(self,action_dim):
        self.action_dim = action_dim

    def __call__(self):
        return np.random.randn(self.action_dim) * 0.01

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.x0 = np.zeros_like(self.mu)
        self.reset()

    def __call__(self):
        dx = self.theta*(self.mu-self.x_prev) + self.sigma*np.random.normal(0.0,1.0,self.mu.shape)
        self.x_prev += dx
        return self.x_prev

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# class OrnsteinUhlenbeckActionNoise:
#     def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = np.zeros_like(self.mu)
#         self.reset()
#
#     def __call__(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
#                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x
#
#     def reset(self):
#         self.x_prev = np.zeros_like(self.mu)
#
#     def __repr__(self):
#         return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)