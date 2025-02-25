import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def equation2D(t,state):
    x1, x2, y1, y2, x1_dot, x2_dot, y1_dot, y2_dot = state
    return [(x1_dot)/mass_x, 
            (x2_dot)/mass_x, 
            (y1_dot)/mass_y, 
            (y2_dot)/mass_y, 
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

# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=10
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Solve the ODE
solution = solve_ivp(equation2D, t_span, initial_conditions, t_eval=t_eval)


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

