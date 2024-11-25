# Проверка пункта 1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def cosin_series_function(t, period=1.0, amplitude=1.0):
  omega = 2 * np.pi / period
  return amplitude * np.cos(omega * t)

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
frequency = 100
amplitude = 5.0
lenght = 4

period = 1 / frequency
n_elements = 20000
n_terms = 10

x_lim_rt = (lenght + 1) / frequency
x_lim_lt = -1 * x_lim_rt
y_lim_rt = amplitude + 1
y_lim_lt = -1 * y_lim_rt

t = np.linspace(-lenght, lenght, n_elements)
cosin = cosin_series_function(t, period=period, amplitude=amplitude)
fourier_series = fourier_series_function(t, cosin_series_function, period=period, 
                                         amplitude=amplitude, n_terms=n_terms)

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].plot(t, cosin, label='Прямоугольный импульс', color='black')
axs[0].plot(t, fourier_series, label='Приближение Фурье', linestyle='--')
axs[0].set_xlim(x_lim_lt, x_lim_rt)
axs[0].set_ylim(y_lim_lt, y_lim_rt)
axs[0].set_title('График прямоугольного импульса и его приближения Фурье')
axs[0].set_xlabel('')
axs[0].set_ylabel('Амплитуда')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t, cosin - fourier_series, color='black')
axs[1].set_xlim(x_lim_lt, x_lim_rt)
axs[1].set_ylim(y_lim_lt, y_lim_rt)
axs[1].set_title('График ошибкок прямоугольного импульса и его приближения Фурье')
axs[1].set_xlabel('Время, c')
axs[1].set_ylabel('Ошибка')
axs[1].grid(True)

plt.tight_layout()
plt.show()


# a)
import numpy as np
import matplotlib.pyplot as plt

# FFT и частоты
fft_values = np.fft.fft(fourier_series)
frequencies = np.fft.fftfreq(n_elements, t[1] - t[0])

# Берем только положительные частоты и нормализуем по числу точек
positive_frequencies = frequencies[:n_elements // 2]
positive_fft_values = np.abs(fft_values[:n_elements // 2]) * 2 / n_elements

# Найдем частоту с максимальной амплитудой
max_amplitude_index = np.argmax(positive_fft_values)
max_frequency = positive_frequencies[max_amplitude_index]
max_amplitude = positive_fft_values[max_amplitude_index]

# Восстановленный сигнал
fft_values_reconstructed = np.fft.ifft(fft_values)

# Границы графиков
max_positive_frequency = np.max(positive_frequencies)
x_spectrum_lim_lt = 0
x_spectrum_lim_rt = max_positive_frequency
y_spectrum_lim_lt = -1
y_spectrum_lim_rt = max_amplitude + 1

# Построение графиков
fig, axs = plt.subplots(1, 2, figsize=(10, 3))

axs[0].plot(t, cosin, label="Начальный сигнал")
axs[0].plot(t, fft_values_reconstructed.real, linestyle='--', label="Востановленный сигнал")
axs[0].set_xlim(x_lim_lt, x_lim_rt)
axs[0].set_ylim(y_lim_lt, y_lim_rt)
axs[0].set_title('Сигнал')
axs[0].set_xlabel('Время, с')
axs[0].set_ylabel('Амплитуда')
axs[0].grid(True)
axs[0].legend(loc='lower right')

axs[1].plot(positive_frequencies, positive_fft_values, color='black', label="Spectrum")
axs[1].set_xlim(x_spectrum_lim_lt, x_spectrum_lim_rt)
axs[1].set_ylim(y_spectrum_lim_lt, y_spectrum_lim_rt)
axs[1].axvline(x=max_frequency, color='orange', linestyle='--', label=f"Частота {max_frequency:.2f} Гц")
axs[1].set_title('Спектр')
axs[1].set_xlabel('Частота, Гц')
axs[1].set_ylabel('Амплитуда')
axs[1].grid(True)

axs[1].annotate(f"a_n = {max_amplitude:.2f}", 
                xy=(max_frequency, max_amplitude),
                xytext=(max_frequency + .5, max_amplitude))

axs[1].legend(loc='upper right')
plt.tight_layout()
plt.show()