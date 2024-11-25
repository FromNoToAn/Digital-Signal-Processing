import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def rectangular_pulse_series_function(t, period=1.0, amplitude=1.0):
  return np.where((t % period) < (period / 2), amplitude, -amplitude)

def fourier_series_function(t, original_function, period=1.0, amplitude=1.0, n_terms=10):
  def a0(period, amplitude):
    result, _ = quad(lambda t: original_function(t, period=period, amplitude=amplitude), 0, period)
    return (2 / period) * result

  def an(n, period, amplitude):
    integrand = lambda t: original_function(t, period=period, amplitude=amplitude) * np.cos(2 * np.pi * n * t / period)
    result, _ = quad(integrand, 0, period)
    return (2 / period) * result

  def bn(n, period, amplitude):
    integrand = lambda t: original_function(t, period=period, amplitude=amplitude) * np.sin(2 * np.pi * n * t / period)
    result, _ = quad(integrand, 0, period)
    return (2 / period) * result
  
  series = np.zeros_like(t)
  series += a0(period, amplitude) / 2
  for n in range(1, n_terms + 1):
    an_coeff = an(n, period, amplitude)
    bn_coeff = bn(n, period, amplitude)
    series += an_coeff * np.cos(2 * np.pi * n * t / period) + bn_coeff * np.sin(2 * np.pi * n * t / period)
  return series

# Задаем параметры
period = 2.0
amplitude = 2.0
lenght = 4

n_elements = 1000
n_terms = 10

x_lim_rt = lenght + 1
x_lim_lt = -1 * x_lim_rt
y_lim_rt = amplitude + 1
y_lim_lt = -1 * y_lim_rt

t = np.linspace(-lenght, lenght, n_elements)
pulse = rectangular_pulse_series_function(t, period=period, amplitude=amplitude)
fourier_series = fourier_series_function(t, rectangular_pulse_series_function, period=period, 
                                         amplitude=amplitude, n_terms=n_terms)

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].plot(t, pulse, label='Прямоугольный импульс')
axs[0].plot(t, fourier_series, label='Приближение Фурье')
axs[0].set_xlim(x_lim_lt, x_lim_rt)
axs[0].set_ylim(y_lim_lt, y_lim_rt)
axs[0].set_title('График прямоугольного импульса и его приближения Фурье')
axs[0].set_xlabel('')
axs[0].set_ylabel('Амплитуда')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t, pulse - fourier_series, color='blue')
axs[1].set_xlim(x_lim_lt, x_lim_rt)
axs[1].set_ylim(y_lim_lt, y_lim_rt)
axs[1].set_title('График ошибкок прямоугольного импульса и его приближения Фурье')
axs[1].set_xlabel('Время, c')
axs[1].set_ylabel('Ошибка')
axs[1].grid(True)

plt.tight_layout()
plt.show()
