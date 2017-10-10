import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

sensor_1_error = sonar1 - range_
sensor_2_error = sonar2 - range_
error1_split = np.array_split(sensor_1_error, 15)
error2_split = np.array_split(sensor_2_error, 15)
range_split = np.array_split(range_, 15)

var1 = []
var2 = []
d = []

for error1, error2, dist in zip(error1_split, error2_split, range_split):
    var1.append(np.var(error1))
    var2.append(np.var(error2))
    d.append(np.mean(dist))

# Fit exponential model to  sensor1 data
coeff = np.polyfit(d, np.log(var1), 1, w=np.sqrt(var1))   # Fit first order model to linearised data
print(coeff)
x = np.linspace(0, 3, num=50)
y = np.exp(coeff[1]) * np.exp(coeff[0]*x) 

# Fit step function to sensor2 data
x2 = np.linspace(0, 3, num=500)
y2 = np.ones(len(x2)) * var2[-1]
y2[:33] = var2[0]

plt.figure()
plt.subplot(1,2,1)
plt.plot(range_, sensor_1_error, '.')
plt.title("Sensor 1 error")
plt.subplot(1,2,2)
plt.plot(range_, sensor_2_error, '.')
plt.title("Sensor 2 error")

plt.figure()
plt.subplot(121)
plt.plot(d, var1, '.', markersize=8, label='Measured error')
plt.plot(x, y, label='Fitted error model')
#plt.title("Sonar 1 error model")
plt.ylabel('Variance')
plt.xlabel("Sensor reading (m)")
plt.legend()

plt.subplot(122)
plt.plot(d, var2, '.', markersize=8, label='Measured error')
plt.plot(x2, y2, label="Fitted error model")
#plt.title("Sonar 2 error model")
plt.ylabel('Variance')
plt.xlabel("Sensor reading (m)")
plt.legend()
plt.show()

