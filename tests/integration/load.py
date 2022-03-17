import numpy as np

a = np.loadtxt('anwser.pout', comments='#', delimiter=',', skiprows=2)

print(a[:,2])
