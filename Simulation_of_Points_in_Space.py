import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, CheckButtons

# Global variables
doppler_effect = True
stored_points = []
B = np.array([0.0, 0.0, 0.0])  # Boost vector
robs = np.array([0.0, 0.0, 0.0])  # Observer coordinates
pmin, pmax = -1000, 1000  # Range for random points, match this with the num_points value

def lorentz_factor(B):
    """Calculate Lorentz factor."""
    beta = np.linalg.norm(B)
    return 1 / np.sqrt(1 - beta**2) if beta < 1 else np.inf

def doppler_shift(B, direction):
    """Calculate Doppler shift for a given direction."""
    beta = np.linalg.norm(B)
    if beta >= 1:
        raise ValueError("Beta should be less than 1 for physical velocities.")
    direction = np.array(direction)
    cos_theta = np.dot(B, direction) / (np.linalg.norm(B) * np.linalg.norm(direction))
    return np.sqrt((1 + beta * cos_theta) / (1 - beta * cos_theta))

def compute_transformed_point(rx, ry, rz, B, robs):
    """Compute the transformed point coordinates based on relativistic effects."""
    rp = np.array([rx, ry, rz])
    a = np.array([4, 8, 9])  # Distance between frames
    Y = lorentz_factor(B)
    p = Y * np.dot(B, rp)
    n = rp + (Y**2 * np.dot(B, rp) * B) / (Y + 1) + a - robs
    w = np.sqrt(Y**2 * (np.dot(B, n) - p)**2 - p**2 + np.dot(n, n))
    ro = Y * (np.dot(B, n) - p) - w
    return Y * ro * B + n + robs

def generate_random_points(B, robs, pmin, pmax, num_points=1000): # num_points can be adjusted
    """Generate and store random points within the specified range."""
    global stored_points
    stored_points = [compute_transformed_point(*np.random.uniform(pmin, pmax, 3), B, robs) for _ in range(num_points)]

def plot_universe(ax):
    """Plot the universe with or without Doppler effect."""
    if not stored_points:
        generate_random_points(B, robs, pmin, pmax)

    points = np.array(stored_points)
    if doppler_effect:
        redshifts = [doppler_shift(B, pt) for pt in points]
    else:
        redshifts = np.ones(len(points))

    ax.clear()
    ax.set_facecolor('black')
    xs, ys, zs = points.T
    ax.scatter(xs, ys, zs, c=redshifts, cmap='coolwarm' if doppler_effect else 'gray', s=5)

    ax.set_axis_off()  # Hide the 3D box and axes
    ax.grid(False)
    ax.set_xlim([pmin, pmax])
    ax.set_ylim([pmin, pmax])
    ax.set_zlim([pmin, pmax])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 1, 0.5]))  # Perspective adjustment
    ax.view_init(elev=robs[2], azim=np.degrees(np.arctan2(robs[1], robs[0])))
    ax.dist = 100  # Set distance from plot

def main():
    """Main function to set up the plot and sliders."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    fig.suptitle('Relativistic Effects of Points in Space', fontsize=16, color='black')

    generate_random_points(B, robs, pmin, pmax)

    plot_universe(ax)

    slider_width, slider_height = 0.10, 0.10
    axcolor = 'white'

    sliders = {
        'beta_x': Slider(plt.axes([0.80, 0.8, slider_width, slider_height], facecolor=axcolor), r'Scaled Velocity $\beta_x$', -0.999, 0.999, valinit=0.0),
        'beta_y': Slider(plt.axes([0.80, 0.65, slider_width, slider_height], facecolor=axcolor), r'Scaled Velocity $\beta_y$', -0.999, 0.999, valinit=0.0),
        'beta_z': Slider(plt.axes([0.80, 0.5, slider_width, slider_height], facecolor=axcolor), r'Scaled Velocity $\beta_z$', -0.999, 0.999, valinit=0.0),
        'obs_x': Slider(plt.axes([0.80, 0.35, slider_width, slider_height], facecolor=axcolor), 'Obs x', pmin, pmax, valinit=0.0),
        'obs_y': Slider(plt.axes([0.80, 0.2, slider_width, slider_height], facecolor=axcolor), 'Obs y', pmin, pmax, valinit=0.0),
        'obs_z': Slider(plt.axes([0.80, 0.05, slider_width, slider_height], facecolor=axcolor), 'Obs z', pmin, pmax, valinit=0.0)
    }

    checkbox_doppler = CheckButtons(plt.axes([0.80, 0.90, 0.10, 0.05], facecolor='white'), ['Doppler Effect'], [doppler_effect])

    def update(val):
        global B, robs
        B = np.array([sliders['beta_x'].val, sliders['beta_y'].val, sliders['beta_z'].val])
        robs = np.array([sliders['obs_x'].val, sliders['obs_y'].val, sliders['obs_z'].val])
        plot_universe(ax)

    def toggle_doppler(label):
        global doppler_effect
        doppler_effect = not doppler_effect
        plot_universe(ax)

    for slider in sliders.values():
        slider.on_changed(update)
    checkbox_doppler.on_clicked(toggle_doppler)

    plt.show()

if __name__ == "__main__":
    main()
