import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
import scipy.signal
from BayesFilters import ParticleFilter
style.use('ggplot')

def main():
    filename = 'training2.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    index, time, range_, *sensorData = data.T
    sensorData = np.array(sensorData).T     # Transpose 
    dt = time[1] - time[0]
    var_w = 9.3e-6

    # Coefficients for linearised infrared sensor models
    ir_coeffs = load_IR_sensor_models()

    filtered_output = []
    ir3_linearised = []
    ir4_linearised = []
    data_output = []

    error = 0

    k = KalmanFilter(var_w, dt)

    for i, (control, *ir_voltages, sonar1, sonar2) in enumerate(sensorData):
        if i == 0:
            pos = 0
            prev_V = ir_voltages

        ir1, ir2, ir3, ir4 = get_ir_linearised(ir_voltages, prev_V, ir_coeffs)
        ir3_linearised.append(ir3)
        ir4_linearised.append(ir4)
        data_vec = [sonar1, sonar2, ir1, ir2, ir3, ir4]
        var_vec = get_sensor_variance(pos)

        data, var_v = fuse_sensors(data_vec, var_vec)
        data_output.append(data)
        k.predict_and_correct(data, control, var_v)
        pos = k.get_estimate()
        filtered_output.append(pos)

        prev_V = ir_voltages 

        error += (range_[i] - pos)**2
   
    # -------- Plotting ----------
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(time, sensorData[:,5])
    ax1.set_title("Sonar sensor 1")

    ax2 = fig.add_subplot(222)
    ax2.plot(time, sensorData[:,6])
    ax2.set_title("Sonar sensor 2")

    ax3 = fig.add_subplot(223)
    ax3.plot(time, ir3_linearised)
    ax3.set_title("IR sensor 3 - linearised")

    ax4 = fig.add_subplot(224)
    ax4.plot(time, ir4_linearised)
    ax4.set_title("IR sensor 4 - linearised")

    #print("Total error = {} for {} process noise".format(error, var_w))
    plt.figure()
    plt.plot(time, range_, linewidth=1.0, label="True position")
    plt.scatter(time, filtered_output, s=3, color="b", label="Filtered output")
    plt.title("Position estimate")
    plt.legend()

    plt.figure()
    plt.plot(time, range_)
    plt.plot(time, data_output)

    plt.tight_layout()
    plt.show()

    print("Error = {:.4f}".format(error))

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

def get_ir_linearised(curr_V, prev_V, coeffs):
    '''
    Linearise infrared sensor function about previous datapoint
    '''
    output_array = []
    for V, prev, coeff in zip(curr_V, prev_V, coeffs):
        a, b, c = coeff
        linearised = (b/(prev - a) - c) - b/(prev - a)**2 * (V - prev)
        nonlin = b / (V - a) - c
        output_array.append(linearised)
    return output_array

def load_IR_sensor_models():
    '''
    Returns the coefficients that define the infrared sensor models
    '''
    coeffs = ((0.10171772, 0.0895424, -0.07046522),
              (-0.27042247, 0.27470697, 0.04806651),
              (0.24727172, 0.22461734, -0.01960771),
              (1.21541481, 1.54949467, -0.00284672))
    return coeffs

def get_sensor_variance(pos):
    '''
    Calculate sensor variances as a function of most recent position estimate
    '''
    # Define sensors useful ranges
    sonar2_cutoff = 0.6
    ir1_cutoff = (0.15, 0.3)
    ir2_cutoff = (0.04, 0.3)
    ir3_cutoff = 1.2
    ir4_cutoff = 1.5

    # Coefficients for exponential variance models
    sonar1_coeff = (0.002977, 2.25556149)
    ir3_coeff = (2.1495e-5, 12.70930048)       # Better results achieved by tweaking model manually. Not sure why
    ir4_coeff =  (0.12185633 ,1.43907183)

    ir3_error_model = ir3_coeff[0] * np.exp(ir3_coeff[1] * pos)
    ir4_error_model = ir4_coeff[0] * np.exp(ir4_coeff[1] * pos)

    var_sonar1 = sonar1_coeff[0] * np.exp(sonar1_coeff[1]*pos)
    var_sonar2 = 2**64 if pos < sonar2_cutoff else 0.015
    var_ir1 = 2**64 if pos < ir1_cutoff[0] or pos > ir1_cutoff[1] else 0.219
    var_ir2 = 2**64 if pos < ir2_cutoff[0] or pos > ir2_cutoff[1] else 0.00103
    var_ir3 = 2**64 if pos > ir3_cutoff else ir3_error_model
    var_ir4 = 2**64 if pos < ir4_cutoff else ir4_error_model
    
    variance_vector = [var_sonar1, var_sonar2, var_ir1, var_ir2, var_ir3, var_ir4]
    return variance_vector

if __name__ == "__main__":
    main()