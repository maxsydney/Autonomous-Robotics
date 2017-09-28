import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize
from matplotlib import style
style.use('ggplot')

# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Fit hyperbolic model
hyperbola = lambda b, x: b[0] + b[1]/(x + b[2])
erf = lambda b: np.linalg.norm(raw_ir3 - hyperbola(b, range_))
xopt = scipy.optimize.fmin(func=erf, x0=[1, 1, 1])

y = hyperbola(xopt, range_)
plt.figure()
plt.plot(range_, raw_ir3)
plt.plot(range_, y)
plt.show()