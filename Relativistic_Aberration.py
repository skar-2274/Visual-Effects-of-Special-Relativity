import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# Mathematical formalism for Relativistic Aberration
def f(phi, B):
    return np.arccos((np.cos(phi) + B) / (1 + B * np.cos(phi)))

# Polar coordinates to Cartesian
def to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

def from_polar(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

# Symmetric relativistic aberration function in Cartesian form
def rnew(r, B):
    cos_phi = (np.cos(r) + B) / (1 + B * np.cos(r))
    cos_phi = np.clip(cos_phi, -1, 1)  # Clip to avoid domain errors
    return np.arccos(cos_phi)

# Image transformation
def apply_transformation(image, B, m=0.8*np.pi):
    width, height = image.size
    x_center, y_center = width / 2, height / 2

    # Generate grid of coordinates
    x = np.linspace(-m, m, width)
    y = np.linspace(-m, m, height)
    xv, yv = np.meshgrid(x, y)

    # Convert to polar coordinates
    r, phi = to_polar(xv, yv)

    # Apply the transformation
    r_transformed = rnew(r, B)
    xv_new, yv_new = from_polar(r_transformed, phi)

    # Convert back to pixel coordinates
    xv_new_pixel = ((xv_new + m) / (2 * m) * width).astype(int)
    yv_new_pixel = ((yv_new + m) / (2 * m) * height).astype(int)

    # Clip values to be within image bounds
    xv_new_pixel = np.clip(xv_new_pixel, 0, width-1)
    yv_new_pixel = np.clip(yv_new_pixel, 0, height-1)

    # Map the pixels from the original image to the new image
    transformed_image = np.array(image)[yv_new_pixel, xv_new_pixel]

    return Image.fromarray(transformed_image)

# Load the image
image_path = 'Stars.jpg'  # Replace with your image path
i = Image.open(image_path)

# Creates a figure and axes for both the plot and the image
fig, (ax_plot, ax_image) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Plot the initial function
phi_values = np.linspace(0, np.pi, 500)
B_initial = 0.0
line, = ax_plot.plot(phi_values, f(phi_values, B_initial), lw=2)
ax_plot.set_xlabel(r'$\phi_s$')
ax_plot.set_ylabel(r'$\phi_o$')
ax_plot.set_title('Relativistic Aberration Function')

# Display the initial image
img_display = ax_image.imshow(i)
ax_image.axis('off')
ax_image.set_title('Transformed Image')

# Slider for B
axB = plt.axes([0.25, 0.1, 0.65, 0.03])
B_slider = Slider(axB, r'Scaled Velocity $\beta$', -0.999, 0.999, valinit=B_initial)

# Update function for the slider
def update(val):
    B = B_slider.val

    # Update the line plot
    line.set_ydata(f(phi_values, B))

    # Update the image
    transformed_image = apply_transformation(i, B)
    img_display.set_data(transformed_image)

    fig.canvas.draw_idle()

B_slider.on_changed(update)

plt.show()
