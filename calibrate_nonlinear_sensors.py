import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.optimize
from matplotlib import style
style.use('ggplot')

def get_ir_range(data, coeff):
    a, b, c = coeff
    nonlin = b / (data-a) - c
    return nonlin

# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Generate range for each sensor
#   - Infrared sensor 1 valid on 0.15 - 1.5m (from spec)
raw_ir1 = raw_ir1[165:317]
range_ir1 = range_[165:317]
time_ir1 = time[165:317]

#   - Infrared sensor 2 valid on 0.04 - 0.3m (from spec)
raw_ir2 = raw_ir2[:317]
range_ir2 = range_[:317]
time_ir2 = time[:317]

#   - Infrared sensor 3 valid on 0.1 - 0.8m (from spec)
raw_ir3 = raw_ir3[:1193]
range_ir3 = range_[:1193]
time_ir3 = time[:1193]

#   - Infrared sensor 4 valid on 1 - 5m (from spec)
raw_ir4 = raw_ir4[934:]
range_ir4 = range_[934:]
time_ir4 = time[934:]

# Fit hyperbolic model
hyperbola = lambda b, x: b[0] + b[1]/(x + b[2])

# Fit hyperbolic model to each sensor
erf = lambda b: np.linalg.norm(raw_ir1 - hyperbola(b, range_ir1))
xopt_1 = scipy.optimize.fmin(func=erf, x0=[1, 1, 1])

erf = lambda b: np.linalg.norm(raw_ir2 - hyperbola(b, range_ir2))
xopt_2 = scipy.optimize.fmin(func=erf, x0=[1, 1, 1])

erf = lambda b: np.linalg.norm(raw_ir3 - hyperbola(b, range_ir3))
xopt_3 = scipy.optimize.fmin(func=erf, x0=[1, 1, 1])

erf = lambda b: np.linalg.norm(raw_ir4 - hyperbola(b, range_ir4))
xopt_4 = scipy.optimize.fmin(func=erf, x0=[1, 0, 1])

y1 = hyperbola(xopt_1, range_ir1)
y2 = hyperbola(xopt_2, range_ir2)
y3 = hyperbola(xopt_3, range_ir3)
y4 = hyperbola(xopt_4, range_ir4)

true_ir1 = get_ir_range(raw_ir1, xopt_1)
true_ir2 = get_ir_range(raw_ir2, xopt_2)
true_ir3 = get_ir_range(raw_ir3, xopt_3)
true_ir4 = get_ir_range(raw_ir4, xopt_4)

# Fit error model to ir sensors
ir1_error = true_ir1 - range_ir1
ir1_error_split = np.array_split(ir1_error, 15)
ir1_range_split = np.array_split(range_ir1, 15)

ir2_error = true_ir2 - range_ir2
ir2_error_split = np.array_split(ir2_error, 15)
ir2_range_split = np.array_split(range_ir2, 15)

ir3_error = true_ir3 - range_ir3
ir3_error_split = np.array_split(ir3_error, 15)
ir3_range_split = np.array_split(range_ir3, 15)

ir4_error = true_ir4 - range_ir4
ir4_error_split = np.array_split(ir4_error, 15)
ir4_range_split = np.array_split(range_ir4, 15)

var_ir1 = []
var_ir2 = []
var_ir3 = []
var_ir4 = []
d1 = []
d2 = []
d3 = []
d4 = []

for error1, error2, error3, error4, dist1, dist2, dist3, dist4 in zip(ir1_error_split, ir2_error_split, ir3_error_split, ir4_error_split, ir1_range_split, ir2_range_split, ir3_range_split, ir4_range_split):
    var_ir1.append(np.var(error1))
    var_ir2.append(np.var(error2))
    var_ir3.append(np.var(error3))
    var_ir4.append(np.var(error4))
    d1.append(np.mean(dist1))
    d2.append(np.mean(dist2))
    d3.append(np.mean(dist3))
    d4.append(np.mean(dist4))

# ----------  Fit exponential model to error data ---------- 
exponential = lambda b, x: b[0] * np.exp(b[1]*x)

ir2_coeff = np.polyfit(d2, var_ir2, 1)

erf = lambda b: np.linalg.norm(var_ir3 - exponential(b, d3))        # This model does not fit very well for some reason
#ir3_coeff = scipy.optimize.fmin(func=erf, x0=[1, 1])
ir3_coeff = np.polyfit(d3, np.log(var_ir3), 1)                      # Fit linear regression model instead

erf = lambda b: np.linalg.norm(var_ir4 - exponential(b, d4))
#ir4_coeff = scipy.optimize.fmin(func=erf, x0=[1, 1])
ir4_coeff = np.polyfit(d4, np.log(var_ir4), 1) 

# ----------  Generate error functions ---------- 
y1 = np.ones(len(d1)) * 0.002

x2 = np.linspace(0.1, 0.3, 50)
y2 = ir2_coeff[1] + ir2_coeff[0]*x2

x3 = np.linspace(0, 1.2, 50)
y3 = np.exp(ir3_coeff[1]) * np.exp(ir3_coeff[0]*0.9*x3)

x4 = np.linspace(1, 3.4, 50)
y4 = np.exp(ir4_coeff[1]) * np.exp(ir4_coeff[0] * x4)

# ----------  Plot fitted models to data ---------- 
# fig1 = plt.figure()
# #fig1.suptitle('Infrared Sensor - Raw Data Fitting', fontsize=10)

# ax1 = fig1.add_subplot(221)
# ax1.plot(range_ir1, raw_ir1, label="Raw data")
# ax1.plot(range_ir1, y1, label="Fitted model")
# ax1.set_title("Infrared sensor 1")
# plt.xlabel("True distance (m)", fontsize=10)
# plt.ylabel("Measured voltage (V)", fontsize=10)
# ax1.legend()

# ax2 = fig1.add_subplot(222)
# ax2.plot(range_ir2, raw_ir2, label="Raw data")
# ax2.plot(range_ir2, y2, label="Fitted model")
# ax2.set_title("Infrared sensor 2")
# plt.xlabel("True distance (m)", fontsize=10)
# plt.ylabel("Measured voltage (V)", fontsize=10)
# ax2.legend()

# ax3 = fig1.add_subplot(223)
# ax3.plot(range_ir3, raw_ir3, label="Raw data")
# ax3.plot(range_ir3, y3, label="Fitted model")
# ax3.set_title("Infrared sensor 3")
# plt.xlabel("True distance (m)", fontsize=10)
# plt.ylabel("Measured voltage (V)", fontsize=10)
# ax3.legend()

# ax4 = fig1.add_subplot(224)
# ax4.plot(range_ir4, raw_ir4, label="Raw data")
# ax4.plot(range_ir4, y4, label="Fitted model")
# ax4.set_title("Infrared sensor 4")
# plt.xlabel("True distance (m)", fontsize=10)
# plt.ylabel("Measured voltage (V)", fontsize=10)
# ax4.legend()

# fig1.tight_layout()

# ---------- Error plots for nonlinear sensors ---------- 
# fig2 = plt.figure()
# fig2.suptitle('Infrared Sensor - Noise', fontsize=10)

# ax1 = fig2.add_subplot(221)
# ax1.plot(range_ir1, raw_ir1 - y1)
# ax1.set_title("Infrared sensor 1", fontsize=10)

# ax2 = fig2.add_subplot(222)
# ax2.plot(range_ir2, raw_ir2 - y2)
# ax2.set_title("Infrared sensor 2", fontsize=10)

# ax3 = fig2.add_subplot(223)
# ax3.plot(range_ir3, raw_ir3 - y3)
# ax3.set_title("Infrared sensor 3", fontsize=10)

# ax4 = fig2.add_subplot(224)
# ax4.plot(range_ir4, raw_ir4 - y4)
# ax4.set_title("Infrared sensor 4", fontsize=10)

# ---------- Plot fitted error models for ir sensors ---------- 
# plt.figure()
# plt.subplot(221)
# plt.title("Infrared sensor 1", fontsize=10)
# plt.plot(d1, var_ir1, '.', markersize=8, label='Measured error')
# plt.plot(d1, y1, label="Fitted error model")
# plt.ylim([-0.001, 0.01])
# plt.xlabel("Distance (m)", fontsize=10)
# plt.ylabel("Variance", fontsize=10)
# plt.legend()

# plt.subplot(222)
# plt.title("Infrared sensor 2", fontsize=10)
# plt.plot(d2, var_ir2, '.', markersize=8, label='Measured error')
# plt.plot(x2, y2, label="Fitted error model")
# plt.ylim([-0.001, 0.01])
# plt.xlabel("Distance (m)", fontsize=10)
# plt.ylabel("Variance", fontsize=10)
# plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.02f'))
# plt.legend()

# plt.subplot(223)
# plt.title("Infrared sensor 3", fontsize=10)
# plt.plot(d3, var_ir3, '.', markersize=8, label='Measured error')
# plt.plot(x3, y3, label="Fitted error model")
# plt.xlabel("Distance (m)", fontsize=10)
# plt.ylabel("Variance", fontsize=10)
# plt.legend()

# plt.subplot(224)
# plt.title("Infrared sensor 4", fontsize=10)
# plt.plot(d4, var_ir4, '.', markersize=8, label='Measured error')
# plt.plot(x4, y4, label="Fitted error model")
# plt.xlabel("Distance (m)", fontsize=10)
# plt.ylabel("Variance", fontsize=10)
# plt.ylim([-0.4, 6])
# plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
# plt.legend()

# plt.tight_layout()

# plt.figure()
# plt.title("IR Sensor 1")
# plt.plot(range_ir1, true_ir1)
# plt.plot(range_ir1, range_ir1)

# plt.figure()
# plt.title("IR Sensor 2")
# plt.plot(range_ir2, true_ir2)
# plt.plot(range_ir2, range_ir2)

# plt.figure()
# plt.title("IR Sensor 3")
# plt.plot(range_ir3, true_ir3)
# plt.plot(range_ir3, range_ir3)

# plt.figure()
# plt.title("IR Sensor 4")
# plt.plot(range_ir4, true_ir4)
# plt.plot(range_ir4, range_ir4)

# print(np.var(true_ir2))
# print(np.var(true_ir2 - range_ir2))

# # Plot squared error as a function of distance
# plt.figure()
# plt.subplot(221)
# plt.plot(range_ir1, (true_ir1 - range_ir1)**2)

# plt.subplot(222)
# plt.plot(range_ir2, (true_ir2 - range_ir2)**2)

# plt.subplot(223)
# plt.plot(range_ir3, (true_ir3 - range_ir3)**2)

# plt.subplot(224)
# plt.plot(range_ir4, (true_ir4 - range_ir4)**2)

plt.show()