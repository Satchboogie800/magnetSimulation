import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import magnetsimulation.plotting as plotting
import magnetsimulation.motion as motion
import magnetsimulation.physics as physics

magnet_position = [[0.,0.]]
m = np.array([[0.,1.]])

ux, uy = plotting.setup_grid(5, 30)
grid_points = np.column_stack([ux.ravel(), uy.ravel()])

B_flat = physics.compute_b_field(grid_points, magnet_position, m, 0.2)
Bx = B_flat[:,0].reshape(ux.shape)
By = B_flat[:,1].reshape(uy.shape)
B_mag = np.sqrt(Bx**2 + By**2)

with np.errstate(divide="ignore", invalid="ignore"):
    Bx_normalized = Bx / B_mag
    By_normalized = By / B_mag

# Replace any NaNs resulting from division by zero back to NaN for matplotlib
Bx_normalized = np.nan_to_num(Bx_normalized)
By_normalized = np.nan_to_num(By_normalized)
# If you want the mask holes back:
Bx_normalized[np.isnan(Bx)] = np.nan
By_normalized[np.isnan(By)] = np.nan

plt.figure()

plt.quiver(ux, uy, Bx_normalized, By_normalized, B_mag, cmap="viridis", angles="xy", scale_units="xy", scale=2.5,
           norm=colors.LogNorm(vmin=1e-10, vmax=1e-7))
plt.title("Magnetic Field Vector Plot")
plt.colorbar(label="Magnetic Field Magnitude")
plt.grid(True)
plt.axis("equal")
plt.show()