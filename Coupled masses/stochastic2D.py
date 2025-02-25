# This is a simple numerical simulation of two coupled point masses, which are subject to Gaussian noise. 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# First, intialise a random seed for reproducibility
np.random.seed(42)

# Now, define the time domain:
t_span = (0, 20)    # From t1 to t2
dt = 0.005          # The time increment
t_eval = np.arange(*t_span, dt)  # Time points for evaluation

# We need random Gaussian noise:
sigma_x1 = .2   # Noise intensity
sigma_x2 = .2
sigma_y1 = .1
sigma_y2 = .4

noise_x1 = sigma_x1 * np.random.normal(0,1, len(t_eval)) / np.sqrt(dt) # Divide by sqrt(dt) for the numerics of a Wiener process!
noise_x2 = sigma_x2 * np.random.normal(0,1, len(t_eval)) / np.sqrt(dt)
noise_y1 = sigma_y1 * np.random.normal(0,1, len(t_eval)) / np.sqrt(dt)
noise_y2 = sigma_y2 * np.random.normal(0,1, len(t_eval)) / np.sqrt(dt)

# For the interpolation with solve_ivp, we need an interpolation:
noise_func_x1 = interp1d(t_eval, noise_x1, kind='linear', fill_value="extrapolate")
noise_func_x2 = interp1d(t_eval, noise_x2, kind='linear', fill_value="extrapolate")
noise_func_y1 = interp1d(t_eval, noise_y1, kind='linear', fill_value="extrapolate")
noise_func_y2 = interp1d(t_eval, noise_y2, kind='linear', fill_value="extrapolate")


def equation2D(t,state):
    x1, x2, y1, y2, x1_dot, x2_dot, y1_dot, y2_dot = state
    return [(x1_dot + noise_func_x1(t))/mass_x, 
            (x2_dot + noise_func_x2(t))/mass_x, 
            (y1_dot + noise_func_y1(t))/mass_y, 
            (y2_dot + noise_func_y2(t))/mass_y, 
            D*(y1-x1) , 
            D*(y2-x2), 
            D*(x1-y1), 
            D*(x2-y2)]

# Initial conditions
mass_x = 1
mass_y = 50

x = [-5,-3]
y = [1.5,0]
vel_x = [0, 0] # this is actually velocity by mass, not just velocity! 
vel_y = [-.05, 0]  # same here!!!
x_dot = [v / mass_x for v in vel_x]
y_dot = [v / mass_y for v in vel_y]

initial_conditions = [*x, *y, *x_dot, *y_dot]
D = 1 # spring constant

# Solve the SDE
# Instead of Euler-Maruyama, we use Runge-Kutter of order 2 or 3 to approximate Euler.
# In combination with our noise integration, this approximates Euler Maruyama for small step sizes, while still keeping the analogy with the deterministic examples. This requires the interpolation of noise.

solution = solve_ivp(equation2D, t_span, initial_conditions, t_eval=t_eval, method='RK23')


# Plot an animation of the solution
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
line, = ax.plot([], [], 'o-')

def animate(i):
    x1, y1, x2, y2, *_ = solution.y[:, i]
    line.set_data([x1, x2], [y1, y2])
    return line,

ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)
plt.show()

