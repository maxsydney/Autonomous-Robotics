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
raw_ir1 = raw_ir1[165:1372]
range_ir1 = range_[165:1372]
time_ir1 = time[165:1372]

#   - Infrared sensor 2 valid on 0.04 - 0.3m (from spec)
raw_ir2 = raw_ir2[:317]
range_ir2 = range_[:317]
time_ir2 = time[:317]

#   - Infrared sensor 3 valid on 0.1 - 0.8m (from spec)
raw_ir3 = raw_ir3[:1372]
range_ir3 = range_[:1372]
time_ir3 = time[:1372]

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

plt.figure()
plt.title("IR Sensor 1")
plt.plot(range_ir1, true_ir1)
plt.plot(range_ir1, range_ir1)

plt.figure()
plt.title("IR Sensor 2")
plt.plot(range_ir2, true_ir2)
plt.plot(range_ir2, range_ir2)

plt.figure()
plt.title("IR Sensor 3")
plt.plot(range_ir3, true_ir3)
plt.plot(range_ir3, range_ir3)

plt.figure()
plt.title("IR Sensor 4")
plt.plot(range_ir4, true_ir4)
plt.plot(range_ir4, range_ir4)

plt.show()

print(xopt_1)
print(xopt_2)