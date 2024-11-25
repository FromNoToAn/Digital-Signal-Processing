import numpy as np
import matplotlib.pyplot as plt

def taylor_exponential(x, n_terms):
  result = 1.0
  term = 1.0
  
  for n in range(1, n_terms + 1):
    term *= x / n
    result += term
  
  return result

# Построение графиков для 5, 6 и 7 степени
x_values = np.linspace(0, 5, 100)
degrees = [5, 6, 7]

for degree in degrees:
  y_values = [taylor_exponential(x, degree) for x in x_values]
  plt.plot(x_values, y_values, label=f'Taylor series (n={degree})')

plt.plot(x_values, np.exp(x_values), 'k--', label='exp(x)')
plt.title('Taylor Series Approximation of e^x')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()