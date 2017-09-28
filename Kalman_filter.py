import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

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

def main():
    filename = 'training2.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    dt = time[1] - time[0]
    var_w = 0.00001
    sonar1_coeff = [0.002977, 2.25556149]
    sonar2_cutoff = 0.6
    filtered_output = []

    k = KalmanFilter(var_w, dt)

    for i, (data1, data2, control) in enumerate(zip(sonar1, sonar2, velocity_command)):
        if len(filtered_output) > 0:
            pos = filtered_output[i-1]
        else:
            pos = 0
        var_data1 = sonar1_coeff[0] * np.exp(sonar1_coeff[1]*pos)
        var_data2 = 2**64 if pos < sonar2_cutoff else 0.015

        data, var_v = fuse_sensors(data1, data2, var_data1, var_data2)
        k.predict_and_correct(data, control, var_v)
        filtered_output.append(k.get_estimate())

    # -------- Plotting ----------
    fig = plt.figure()
    ax1 = fig.add_subplot(222)
    ax1.plot(time, sonar1)
    ax1.set_title("Sonar sensor 1")

    ax2 = fig.add_subplot(224)
    ax2.plot(time, sonar2)
    ax2.set_title("Sonar sensor 2")

    ax3 = fig.add_subplot(121)
    ax3.plot(time, range_, linewidth=1.0, label="True position")
    ax3.scatter(time, filtered_output, s=3, color="b", label="Filtered output")
    ax3.set_title("Position estimate")
    ax3.legend()

    plt.tight_layout()
    plt.show()

def fuse_sensors(data1, data2, var1, var2):
    '''
    Fuse sensor data for kalman filter
    '''
    data = ((1/var1) * data1 + (1/var2) * data2) / (1/var1 + 1/var2)
    var_v = (var1 * var2) / (var1 + var2)
    return data, var_v

if __name__ == "__main__":
    main()