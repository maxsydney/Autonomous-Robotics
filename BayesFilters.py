import numpy as np 

class KalmanFilter(object):

    def __init__(self, var_w, dt):
        self.var_w = var_w
        self.x_post = 0
        self.var_x_post = 1.0
        self.dt = dt

    def predict_and_correct(self, data, control, var_v):
        '''
        Perform one iteration of the Kalman filter with new data
        '''
        # Predict step
        x_prior = self.x_post + control * self.dt
        var_x_prior = self.var_x_post + self.var_w

        # Correct step
        z = data
        K = var_x_prior / (var_v + var_x_prior)
        self.x_post = x_prior + K * (z - x_prior)
        self.var_x_post = (1 - K) * var_x_prior
    
    def get_estimate(self):
        '''
        Returns current estimate from Kalman filter
        '''
        return self.x_post

class ParticleFilter(object):
    pass
    