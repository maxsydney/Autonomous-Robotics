import numpy as np
import scipy.stats
import bisect

class KalmanFilter(object):
    """ Simple implementation for a one dimensional Kalman filter with additive noise
    sensor and motion models """

    def __init__(self, var_w, dt):
        self.var_w = var_w
        self.x_post = 0
        self.var_x_post = 1.0
        self.dt = dt

    def predict_and_correct(self, data, control, var_v):
        """Perform one iteration of the Kalman filter with new data"""
        # Predict step
        x_prior = self.x_post + control * self.dt
        var_x_prior = self.var_x_post + self.var_w

        # Correct step
        z = data
        K = var_x_prior / (var_v + var_x_prior)
        self.x_post = x_prior + K * (z - x_prior)
        self.var_x_post = (1 - K) * var_x_prior
    
    def get_estimate(self):
        """Returns current estimate from Kalman filter"""
        return self.x_post

class ParticleFilter(object):
    """ Simple implementation for a one dimensional particle filter with additive noise
    sensor and motion models """
    
    def __init__(self, var_W, dt, range_, n_particles):
        self.var_W = var_W
        self.dt = dt
        self.n_particles = n_particles
        self.range = range_
        self.posterior = 0.1
        self.create_particles(init=0)
        self.n_resamples = 0

    def create_particles(self, init=0):
        """Creates initial particle and weight vectors"""
        if init:
            self.particles = np.random.randn(self.n_particles) + self.posterior
        else:
            self.particles = np.random.uniform(0, self.range, self.n_particles)
        self.weights = np.ones(self.n_particles) * 1/self.n_particles

    def motion_model(self, control):
        return control * self.dt + np.random.randn(self.n_particles)*self.var_W
    
    def predict_and_correct(self, data, control, var_v):
        """Execute one iteration of the particle filter"""
        # Predict step
        self.particles += self.motion_model(control)

        # Correct step
        self.update_weights(data, var_v)
        if self.is_degenerate():
            self.resample()
            self.n_resamples += 1
        
        # Compute state estimate
        self.posterior = np.average(self.particles, weights=self.weights)

    def update_weights(self, data, var_v):
        """Update weights based on a gaussian distribution"""
        var_gain = 250  # Amplify variance to avoid killing off too many particles
        self.weights *= scipy.stats.norm(data, var_v * var_gain).pdf(self.particles)
        self.weights += 1.e-300
        self.weights /= sum(self.weights)
    
    def get_estimate(self):
        return self.posterior

    def resample(self):
        """Resample particles in proportion to their weights.

        Particles and weights should be arrays, and will be updated in place."""

        cum_weights = np.cumsum(self.weights)
        cum_weights /= cum_weights[-1]

        new_particles = []
        for _ in self.particles:
            # Copy a particle into the list of new particles, choosing based
            # on weight
            m = bisect.bisect_left(cum_weights, np.random.uniform(0, 1))
            p = self.particles[m]
            new_particles.append(p)

        # Replace old particles with new particles
        for m, p in enumerate(new_particles):
            self.particles[m] = p

        # Reset weights
        self.weights[:] = 1 / self.n_particles

    def is_degenerate(self):
        """Return true if the particles are degenerate and need resampling."""
        w = self.weights/np.sum(self.weights)
        return 1/np.sum(w**2) < 0.5*len(w)

class Particle(object):
    """Class describing a particle for the particle filter"""

    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class ParticleFilter_2D(ParticleFilter):
    """ Extension of the particle filter for localization of a wheeled robot using
    stochastic/odometry motion models """
    
    def create_particles(self, init=0):
        self.particles = []
        for _ in self.n_particles:
            x = np.random.uniform(0, self.range_[0])
            y = np.random.uniform(0, self.range_[1])
            theta = np.random.uniform(0, 2*np.pi)
            self.particles.append(Particle(x, y, theta, 1/self.n_particles))

    def motion_model(self, control):
        pass
    