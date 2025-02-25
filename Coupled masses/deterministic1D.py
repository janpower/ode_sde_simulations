import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def equation1D(t, state):
    x1, x2, x1_dot, x2_dot = state
    return [x1_dot, x2_dot, D * (x2 - x1), D * (x1 - x2)]


# Initial conditions
initial_conditions = [1, 1, -.3, .299]
D = 0.1

# Time span for the simulation
t_span = (0, 100)  # From t=0 to t=10
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Solve the ODE
solution = solve_ivp(equation1D, t_span, initial_conditions, t_eval=t_eval)


# Plot an animation of the solution
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 1)
line, = ax.plot([], [], 'o-')

def animate(i):
    x1, x2, *_ = solution.y[:, i]
    line.set_data([x1, x2], [0,0])
    return line,

ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)
plt.show()

