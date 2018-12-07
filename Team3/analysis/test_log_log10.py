import numpy as np
import matplotlib.pyplot as plt

z = range(1,100)

x = np.log(z)
y = np.log10(z)
plt.plot(x, y, color = 'blue', marker = "*")

plt.title("log10 ln")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
