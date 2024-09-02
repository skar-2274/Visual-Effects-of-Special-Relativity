import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image

# Mathematical formalism for Relativistic Aberration
def f(phi, B):
    return np.arccos((np.cos(phi) + B) / (1 + B * np.cos(phi)))

def to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

def from_polar(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def rnew(r, B):
    cos_phi = (np.cos(r) + B) / (1 + B * np.cos(r))
    cos_phi = np.clip(cos_phi, -1, 1)
    return np.arccos(cos_phi)

def apply_transformation(image, B, m=0.8*np.pi):
    width, height = image.size
    x_center, y_center = width / 2, height / 2

    x = np.linspace(-m, m, width)
    y = np.linspace(-m, m, height)
    xv, yv = np.meshgrid(x, y)
    
    r, phi = to_polar(xv, yv)

    r_transformed = rnew(r, B)
    xv_new, yv_new = from_polar(r_transformed, phi)

    xv_new_pixel = ((xv_new + m) / (2 * m) * width).astype(int)
    yv_new_pixel = ((yv_new + m) / (2 * m) * height).astype(int)

    xv_new_pixel = np.clip(xv_new_pixel, 0, width-1)
    yv_new_pixel = np.clip(yv_new_pixel, 0, height-1)

    transformed_image = np.array(image)[yv_new_pixel, xv_new_pixel]

    return Image.fromarray(transformed_image)

image_path = 'Stars.jpg'  # Replace with your image path
i = Image.open(image_path)

fig, (ax_plot, ax_image) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

phi_values = np.linspace(0, np.pi, 500)
B_initial = 0.0
line, = ax_plot.plot(phi_values, f(phi_values, B_initial), lw=2)
ax_plot.set_xlabel(r'$\phi_s$')
ax_plot.set_ylabel(r'$\phi_o$')
ax_plot.set_title('Relativistic Aberration Function')

img_display = ax_image.imshow(i)
ax_image.axis('off')
ax_image.set_title('Transformed Image')

axB = plt.axes([0.25, 0.1, 0.65, 0.03])
B_slider = Slider(axB, r'Scaled Velocity $\beta$', -0.999, 0.999, valinit=B_initial)

def update(val):
    B = B_slider.val

    line.set_ydata(f(phi_values, B))

    transformed_image = apply_transformation(i, B)
    img_display.set_data(transformed_image)

    fig.canvas.draw_idle()

B_slider.on_changed(update)

plt.show()
