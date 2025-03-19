import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def random_walk(N, steps, dh):
    positions = np.zeros(N)
    history = np.zeros((steps, N))
    
    for t in range(steps):
        steps_random = np.random.uniform(-dh, dh, N)
        positions += steps_random  
        history[t] = positions
    
    return history

def linear_func(x, a, b):
    return a * x + b

def calculate_coefficients(data, steps):
    x_mean = np.mean(data, axis=1)
    x2_mean = np.mean(data**2, axis=1)
    popt_x, _ = curve_fit(linear_func, np.arange(steps), x_mean)
    popt_x2, _ = curve_fit(linear_func, np.arange(steps), x2_mean)
    
    return x_mean, x2_mean, popt_x, popt_x2

def find_stable_step_x(values, threshold):
    for i in range(1, len(values)):
        if np.abs(values[i] - values[i-1]) / np.abs(values[i]) <= threshold:
            return i
    return len(values)

def find_stable_step_x2(values, threshold):
    for i in range(1, len(values)):
        if np.abs(values[i] - values[i-1]) / np.abs(values[i]) <= threshold:
            if np.abs(values[i+1] - values[i]) / np.abs(values[i]) <= threshold:
                return i + 1
    return len(values)

N = 1000
steps = 2000
dh = 1.0

data = random_walk(N, steps, dh)

x_mean, x2_mean, popt_x, popt_x2 = calculate_coefficients(data, steps)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(np.arange(steps), x_mean, label='<x>')
plt.plot(np.arange(steps), linear_func(np.arange(steps), *popt_x), 'r--', label='Linear fit')
plt.xlabel('Steps')
plt.ylabel('<x>')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.arange(steps), x2_mean, label='<x²>')
plt.plot(np.arange(steps), linear_func(np.arange(steps), *popt_x2), 'r--', label='Linear fit')
plt.xlabel('Steps')
plt.ylabel('<x²>')
plt.legend()

plt.tight_layout()
plt.show()

thresholds = [0.01, 0.001, 0.0001]
for threshold in thresholds:
    stable_step_x = find_stable_step_x(x_mean, threshold)
    stable_step_x2 = find_stable_step_x2(x2_mean, threshold)
    print()
    print(f"For accuracy {threshold*100}%:")
    print(f"  <x> stabilizes at step {stable_step_x}")
    print(f"  <x²> stabilizes at step {stable_step_x2}")