import numpy as np 
import matplotlib.pyplot as plt
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

# Fit error model to ir3
ir3_error = true_ir3 - range_ir3
ir3_error_split = np.array_split(ir3_error, 10)
ir3_range_split = np.array_split(range_ir3, 10)

ir4_error = true_ir4 - range_ir4
ir4_error_split = np.array_split(ir4_error, 10)
ir4_range_split = np.array_split(range_ir4, 10)

var_ir3 = []
var_ir4 = []
d3 = []
d4 = []

for error3, error4, dist3, dist4 in zip(ir3_error_split, ir4_error_split, ir3_range_split, ir4_range_split):
    var_ir3.append(np.var(error3))
    var_ir4.append(np.var(error4))
    d3.append(np.mean(dist3))
    d4.append(np.mean(dist4))

exponential = lambda b, x: b[0] * np.exp(b[1]*x)

erf = lambda b: np.linalg.norm(var_ir3 - exponential(b, d3))        # This model does not fit very well for some reason
ir3_coeff = scipy.optimize.fmin(func=erf, x0=[1, 1])
ir3_coeff = np.polyfit(d3, np.log(var_ir3), 1)                      # Fit linear regression model instead

erf = lambda b: np.linalg.norm(var_ir4 - exponential(b, d4))
ir4_coeff = scipy.optimize.fmin(func=erf, x0=[1, 1])

x = np.linspace(0, 1.2, 50)
y = 12.70930048*np.exp(ir3_coeff[0]*x)

plt.figure()
plt.plot(d3, var_ir3)
plt.plot(x, y)

plt.figure()
plt.plot(d4, var_ir4)
plt.plot(d4, ir4_coeff[0] * np.exp(ir4_coeff[1]*d4))

print(ir3_coeff)
print(ir4_coeff)
# fig1 = plt.figure()
# fig1.suptitle('Infrared Sensor - Raw Data Fitting', fontsize=10)

# ax1 = fig1.add_subplot(221)
# ax1.plot(range_ir1, raw_ir1)
# ax1.plot(range_ir1, y1)
# ax1.set_title("Infrared sensor 1", fontsize=10)

# ax2 = fig1.add_subplot(222)
# ax2.plot(range_ir2, raw_ir2)
# ax2.plot(range_ir2, y2)
# ax2.set_title("Infrared sensor 2", fontsize=10)

# ax3 = fig1.add_subplot(223)
# ax3.plot(range_ir3, raw_ir3)
# ax3.plot(range_ir3, y3)
# ax3.set_title("Infrared sensor 3", fontsize=10)

# ax4 = fig1.add_subplot(224)
# ax4.plot(range_ir4, raw_ir4)
# ax4.plot(range_ir4, y4)
# ax4.set_title("Infrared sensor 4", fontsize=10)

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
# plt.figure()
# plt.plot(true_ir2 - range_ir2)

# plt.figure()
# plt.plot(true_ir1 - range_ir1)



plt.show()