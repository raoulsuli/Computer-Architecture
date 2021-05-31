import matplotlib.pyplot as plt
import numpy as np

plt.title('Grafice Analiza Comparativa')
plt.xlabel('N')
plt.ylabel('Timp rulare (s)')

x = np.linspace(400, 1400, num=6)
types = ['neopt', 'blas', 'opt']
runtimes = [[0.747984, 2.834365, 7.538070, 16.405756, 29.194471, 50.236938], [0.037375, 0.140533, 0.238549, 0.432290, 0.745313, 1.168161], [0.299155, 0.969092, 2.448821, 4.578099, 8.663284, 14.175742]]

for i in range(3):
    plt.plot(x, runtimes[i], label=types[i])

plt.ylim(top=17)
plt.legend()
plt.show()