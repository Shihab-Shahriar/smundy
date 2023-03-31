import numpy as np

N = 3200

data = np.genfromtxt('mat.csv', delimiter=',', dtype='float')
A = data[:N,:]
x = data[N]
y = A@x 
print(y.mean())

y_a = np.genfromtxt('kfile.csv', delimiter=',')
print(y_a[0].mean())
print(y_a[1].mean())