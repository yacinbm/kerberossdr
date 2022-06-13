from scipy.stats import linregress
import numpy as np

data = np.loadtxt("verif.csv", delimiter=',')
slope, intercept, r_value, p_value, std_err = linregress(data[0,:], data[1,:])

print(f"R_squared: {r_value}")