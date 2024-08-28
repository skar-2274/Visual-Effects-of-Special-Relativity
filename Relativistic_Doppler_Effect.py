import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, CheckButtons
from matplotlib.colors import LinearSegmentedColormap

# Relativistic Aberration
def aberration(B, phi):
    return np.arccos((np.cos(phi) + B) / (1 + B * np.cos(phi)))

# Doppler shift
def doppler_shift(B, phi):
    return np.sqrt(1 - B**2) / (1 + B * np.cos(phi))

# Zero function
def zero_shift(B):
    return np.arccos((1 - np.sqrt(1 - B**2)) / B)

# Creates a custom colormap for Doppler shift (blue to white to red)
cmap = LinearSegmentedColormap.from_list("Redshift", ["blue", "white", "red"])

# Function to normalize the Doppler shift to range 0-1 where 0.5 is no shift (white)
def normalize_doppler(doppler_value):
    return np.clip(doppler_value - 0.5, 0, 1)

# Function to create lines in the specified direction
def create_lines(B, n, color_lines, direction='up'):
    lines = []
    for i in range(n + 1 if direction == 'down' else n):
        phi = np.pi / n * i
        alpha = aberration(B, phi)
        x = [np.cos(alpha), 2 * np.cos(alpha)]
        y = [np.sin(alpha), 2 * np.sin(alpha)] if direction == 'up' else [-np.sin(alpha), -2 * np.sin(alpha)]

        color = 'white'
        if color_lines:
            doppler_value = doppler_shift(B, phi)
            normalized_value = normalize_doppler(doppler_value)
            color = cmap(normalized_value)

        lines.append((x, y, color))
    return lines

# Function to plot the arrows
def draw_arrow(B):
    arrow_coords = [[-0.1, -0.1], [0.1, 0], [-0.1, 0.1]] if B >= 0 else [[0.1, -0.1], [-0.1, 0], [0.1, 0.1]]
    return Polygon(arrow_coords, edgecolor='black', facecolor='yellow')

# Initial values
B_init = 0.0
n_init = 10
color_lines_init = True

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.1, bottom=0.3)
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

def plot_graph(B, n, color_lines):
    ax.clear()
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    for (x, y, color) in create_lines(B, n, color_lines, direction='up'):
        ax.plot(x, y, color=color, linewidth=2)

    for (x, y, color) in create_lines(B, n, color_lines, direction='down'):
        ax.plot(x, y, color=color, linewidth=2)

    ax.add_patch(draw_arrow(B))

    label_text = "Direction with no Doppler shift = "
    label_text += "Undetermined" if B == 0 else f"{np.degrees(zero_shift(B)):.4f} deg"

    ax.text(0, 2.3, label_text, color='grey', fontsize=12,
            ha='center', va='center',
            bbox=dict(facecolor='none', edgecolor='none', alpha=0.7, pad=10))
    ax.set_title('Relativistic Doppler Effect', color='black', fontsize=14)
    fig.canvas.draw_idle()

# Draw the initial plot
plot_graph(B_init, n_init, color_lines_init)

ax_B = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_B = Slider(ax_B, r'Scaled Velocity $\beta$', -0.999, 0.999, valinit=B_init)

ax_n = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_n = Slider(ax_n, 'Number of Light Rays', 10, 100, valinit=n_init, valstep=1)

ax_col = plt.axes([0.8, 0.1, 0.15, 0.15])
checkbox_col = CheckButtons(ax_col, ['Show Doppler Effect'], [color_lines_init])

def update(val):
    B = slider_B.val
    n = int(slider_n.val)
    color_lines = checkbox_col.get_status()[0]
    plot_graph(B, n, color_lines)

slider_B.on_changed(update)
slider_n.on_changed(update)
checkbox_col.on_clicked(update)

plt.show()
