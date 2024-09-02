import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# Function to compute lines for cube
def compute_line(rx, ry, rz, ox, oy, oz, Bx, By, Bz, robsx, robsy, robsz, s_values):
    B = np.array([Bx, By, Bz]) # Relativistic Boost Components
    rp = np.array([rx, ry, rz]) # Spatial components of the spacetime coordinates
    robs = np.array([robso, robsx, robsy, robsz]) # Spatial components of the observers coordinates
    o = np.array([ox, oy, oz]) / np.sqrt(ox**2 + oy**2 + oz**2) # Direction vector
    Y = 1 / np.sqrt(1 - np.dot(B, B)) # Lorentz factor

    p = Y * np.dot(B, rp) - robs[0] # Abbreviation
    n = rp + Y**2 / (Y + 1) * np.dot(B, rp) * B + a - robs[1:] # Abbreviation
    u = n + B * (-Y * p + Y**2 / (Y + 1) * np.dot(B, n)) # Abbreviation
    v = np.dot(u, o) # Abbreviation
    lines = []

    for s in s_values:
        w = np.sqrt((Y**2 * (np.dot(B, n) - p)**2 - p**2 + np.dot(n, n))) # Abbreviation
        wline = np.sqrt(w**2 + 2 * s * v + s**2) # Abbreviation
        rlo = Y * (np.dot(B, n) - p) - wline # Starting Point of a line
        rline = Y * rlo * B + n + robs[1:] + s * o + (Y**2 * s) / (Y + 1) * np.dot(B, o) * B # Equation of a line
        lines.append(rline)

    return np.array(lines)

# Initial parameters
B = np.array([0.0, 0.0, 0.0])
robs = np.array([0.0, 0.0, 0.0, 0.0]) # Observers coordinates
a = np.array([1, 2, 3]) # Distance between two frames
smax = 1000 # Size of the cube
s_values = np.linspace(0, smax, 100)
robso = 0.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Function to generate cube lines
def generate_cube_lines(smax, B, robs, s_values):
    lines = []
    # Define start and end points for each edge of the cube
    edges = [
        [(0, 0, 0), (smax, 0, 0)], [(0, 0, 0), (0, smax, 0)], [(0, 0, 0), (0, 0, smax)],
        [(smax, 0, 0), (smax, smax, 0)], [(smax, 0, 0), (smax, 0, smax)], [(0, smax, 0), (smax, smax, 0)],
        [(0, smax, 0), (0, smax, smax)], [(0, 0, smax), (smax, 0, smax)], [(0, 0, smax), (0, smax, smax)],
        [(smax, smax, 0), (smax, smax, smax)], [(smax, 0, smax), (smax, smax, smax)], [(0, smax, smax), (smax, smax, smax)]
    ]

    for start, end in edges:
        start = np.array(start)
        end = np.array(end)
        direction = end - start
        lines.append(compute_line(*start, *direction, *B, *robs, s_values))

    return lines

def plot_cube():
    ax.clear()
    line_segments = generate_cube_lines(smax, B, robs[1:], s_values)

    for segment in line_segments:
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Relativistic Cube Simulation')  # Title for the plot

plot_cube()

slider_width = 0.10
slider_height = 0.10
ax_slider_Bx = plt.axes([0.75, 0.8, slider_width, slider_height])
ax_slider_By = plt.axes([0.75, 0.65, slider_width, slider_height])
ax_slider_Bz = plt.axes([0.75, 0.5, slider_width, slider_height])
ax_slider_robsx = plt.axes([0.75, 0.35, slider_width, slider_height])
ax_slider_robsy = plt.axes([0.75, 0.2, slider_width, slider_height])
ax_slider_robsz = plt.axes([0.75, 0.05, slider_width, slider_height])

slider_Bx = Slider(ax_slider_Bx, r'$\beta_x$', -0.999, 0.999, valinit=B[0], orientation='horizontal')
slider_By = Slider(ax_slider_By, r'$\beta_y$', -0.999, 0.999, valinit=B[1], orientation='horizontal')
slider_Bz = Slider(ax_slider_Bz, r'$\beta_z$', -0.999, 0.999, valinit=B[2], orientation='horizontal')
slider_robsx = Slider(ax_slider_robsx, 'Obs x', -1000, 1000, valinit=robs[1], orientation='horizontal')
slider_robsy = Slider(ax_slider_robsy, 'Obs y', -1000, 1000, valinit=robs[2], orientation='horizontal')
slider_robsz = Slider(ax_slider_robsz, 'Obs z', -1000, 1000, valinit=robs[3], orientation='horizontal')

def update(val):
    global B, robs
    B = np.array([slider_Bx.val, slider_By.val, slider_Bz.val])
    robs = np.array([0, slider_robsx.val, slider_robsy.val, slider_robsz.val])

    plot_cube()

slider_Bx.on_changed(update)
slider_By.on_changed(update)
slider_Bz.on_changed(update)
slider_robsx.on_changed(update)
slider_robsy.on_changed(update)
slider_robsz.on_changed(update)

plt.show()
