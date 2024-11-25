import numpy as np
from scipy.integrate import nquad
import matplotlib.pyplot as plt

# def integrand(z, y, x):
#   return (x + y + z) / np.sqrt(2*x**2 + 4*y**2 + 5*z**2)

# def z_limits(y, x):
#   return [0, np.sqrt(1 - x**2 - y**2)]

# def y_limits(x):
#   return [0, np.sqrt(1 - x**2)]

# def x_limits():
#   return [0, 1]

# result, error = nquad(integrand, [z_limits, y_limits, x_limits])

# print(f"Результат интегрирования: {result}")


def rectangular_pulse_integral(start, end, amplitude):
  width = end - start
  area = width * amplitude
  return area

start = 0       # Начало импульса
end = 1         # Конец импульса
amplitude = 2   # Амплитуда импульса

integral_value = rectangular_pulse_integral(start, end, amplitude)
print(f"Интеграл (площадь прямоугольника) = {integral_value}")

lt_shift = start - 1
rt_shift = end + 1
x_values = [lt_shift, start, start, end, end, rt_shift]
y_values = [0, 0, amplitude, amplitude, 0, 0]

# Построение графика
plt.plot(x_values, y_values, color='black', label=f'Rectangular Pulse (Amplitude={amplitude})')
plt.fill_between(x_values, y_values, color='black', alpha=0.1)
plt.xlim(lt_shift, rt_shift)
plt.ylim(0, amplitude + 1)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Rectangular Pulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()