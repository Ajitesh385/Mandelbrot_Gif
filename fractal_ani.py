import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


plt.switch_backend('Agg')  # prevents Qt-related __del__ errors

width, height = 800, 600
max_iter = 100
frames = 60
x_center, y_center = -0.743643887037151, 0.131825904205330
zoom_start, zoom_end = 1.0, 0.002

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    xs = np.linspace(xmin, xmax, width)
    ys = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(xs, ys[::-1])
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    div_time = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)
    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.abs(Z) > 2.0
        newly_escaped = escaped & mask
        div_time[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    with np.errstate(divide='ignore', invalid='ignore'):
        zn = Z
        m = div_time > 0
        div_time = div_time.astype(float)
        div_time[m] = div_time[m] + 1 - np.log(np.log(np.abs(zn[m])))/np.log(2)
    div_time[div_time == 0] = max_iter
    return div_time

def extent_for_zoom(center_x, center_y, zoom_factor):
    span = 3.1 * zoom_factor
    xmin = center_x - span/2
    xmax = center_x + span/2
    ymin = center_y - (span * height/width)/2
    ymax = center_y + (span * height/width)/2
    return xmin, xmax, ymin, ymax

# Precompute all frames to speed up animation
zoom_factors = np.geomspace(zoom_start, zoom_end, frames)
frames_data = []
for zi in zoom_factors:
    xmin, xmax, ymin, ymax = extent_for_zoom(x_center, y_center, zi)
    img = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    frames_data.append((img, (xmin, xmax, ymin, ymax)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_axis_off()
im = ax.imshow(frames_data[0][0], origin='lower', extent=frames_data[0][1])

def update(frame_idx):
    img, ext = frames_data[frame_idx]
    im.set_data(img)
    im.set_extent(ext)
    return (im,)

anim = animation.FuncAnimation(fig, update, frames=frames, interval=60, blit=True)

# Save as GIF using Pillow (safe even without ffmpeg/imagemagick)
anim.save('mandelbrot_zoom.gif', writer='pillow', fps=20)
print('Saved animation to mandelbrot_zoom.gif')