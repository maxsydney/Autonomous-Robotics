import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
from collections import namedtuple
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
    filename = 'training1.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    dt = time[1] - time[0]
    var_w = 9.3e-6
    sonar1_coeff = [0.002977, 2.25556149]
    ir_coeffs = load_IR_sensor_models()
    sonar2_cutoff = 0.6
    ir3_cutoff = 0.11
    ir4_cutoff = 1.5
    filtered_output = []
    ir3_linearised = []
    ir4_linearised = []

    k = KalmanFilter(var_w, dt)

    for i, (data1, data2, data3, data4, control) in enumerate(zip(sonar1, sonar2, raw_ir3, raw_ir4, velocity_command)):
        if i > 0:
            pos = filtered_output[i-1]
            prev_V = raw_ir3[i-1]
        else:
            pos = 0
            prev_V3 = data3
            prev_V4 = data4
        var_data1 = sonar1_coeff[0] * np.exp(sonar1_coeff[1]*pos)
        var_data2 = 2**64 if pos < sonar2_cutoff else 0.015
        var_data3 = 2**64 if pos > ir3_cutoff else 0.501
        var_data4 = 2**64 if pos < ir4_cutoff else 1.97
        data3 = get_ir_linearised(data3, prev_V3, ir_coeffs["ir3"])
        data4 = get_ir_linearised(data4, prev_V4, ir_coeffs["ir4"])
        ir3_linearised.append(data3)
        ir4_linearised.append(data4)
        data_vec = [data1, data2, data3, data4]
        var_vec = [var_data1, var_data2, var_data3, var_data4]

        data, var_v = fuse_sensors(data_vec, var_vec)
        k.predict_and_correct(data, control, var_v)
        filtered_output.append(k.get_estimate())

    # -------- Plotting ----------
    fig = plt.figure()
    ax1 = fig.add_subplot(232)
    ax1.plot(time, sonar1)
    ax1.set_title("Sonar sensor 1")

    ax2 = fig.add_subplot(233)
    ax2.plot(time, sonar2)
    ax2.set_title("Sonar sensor 2")

    ax3 = fig.add_subplot(235)
    ax3.plot(time, ir3_linearised)
    ax3.set_title("IR sensor 3 - linearised")

    ax4 = fig.add_subplot(236)
    ax4.plot(time, ir4_linearised)
    ax4.set_title("IR sensor 4 - linearised")

    #print("Total error = {} for {} process noise".format(error, var_w))
    ax5 = fig.add_subplot(131)
    ax5.plot(time, range_, linewidth=1.0, label="True position")
    ax5.scatter(time, filtered_output, s=3, color="b", label="Filtered output")
    ax5.set_title("Position estimate")
    ax5.legend()

    plt.tight_layout()
    plt.show()

def fuse_sensors(data_vec, var_vec):
    '''
    Fuse sensor data for kalman filter
    '''
    num = denom = 0 
    for data,var in zip(data_vec,var_vec):
        num += (1 / var) * data
        denom += 1 / var
    data = num / denom
    var_v = 1 / denom
    return data, var_v

def get_ir_linearised(data3, prev_V, coeff):
    '''
    Linearise infrared sensor function about previous datapoint
    '''
    a, b, c = coeff
    linearised = (b/(prev_V - a) - c) - b/(prev_V - a)**2 * (data3 - prev_V)
    nonlin = b / (data3-a) - c
    return linearised

def load_IR_sensor_models():
    '''
    Returns the coefficients that define the infrared sensor models
    '''
    coeffs = {"ir1": [0.10171772, 0.0895424, -0.07046522],
              "ir2": [-0.27042247, 0.27470697, 0.04806651],
              "ir3": [0.24727172, 0.22461734, -0.01960771],
              "ir4": [1.21541481, 1.54949467, -0.00284672]}
    return coeffs

if __name__ == "__main__":
    main()