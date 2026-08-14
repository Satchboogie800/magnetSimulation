import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from magnetsimulation.motion import get_magnet_position
import magnetsimulation.plotting as plotting
import magnetsimulation.motion as motion
import magnetsimulation.physics as physics

max_distance = 5 # [m]
resolution = 25 # points per dimension
X,Y,Z = plotting.setup_grid(max_distance, resolution)
rr_grid = np.stack([X, Y, Z], axis=-1)
print(rr_grid.shape)

mu_0 = 4 * np.pi * 10**-7
eps_0 = 8.854e-12
v_x = 0.5

t_snapshot = 1.025
dt_num = 0.005

B_current, dB_dt, E_current = physics.get_field_at_time(t=t_snapshot, grid=rr_grid, r_grid=rr_grid)
B_mag = np.linalg.norm(B_current, axis=-1, keepdims=True)
B_norm = B_current / B_mag
B_x = B_norm[:, :, 0]
B_y = B_norm[:, :, 1]
B_z = B_norm[:, :, 2]

E_mag = np.linalg.norm(E_current, axis=-1, keepdims=True)
E_norm = E_current / E_mag
E_x = E_norm[:, :, 0]
E_y = E_norm[:, :, 1]
E_z = E_norm[:, :, 2]
print(E_mag[0:20])
fig, axes = plt.subplots(1, 2)
q1 = axes[0].quiver(X, Z, B_x, B_z, B_mag, scale=3, scale_units='inches', norm=colors.LogNorm(vmin=1e-7, vmax=1e-4))
axes[0].set_title(f'Magnetic Field $\mathbf{{B}}$ Quiver (t = {t_snapshot}s, y=1m)')
axes[0].set_xlabel('X Position (m)')
axes[0].set_ylabel('Z Position (m)')
axes[0].set_aspect('equal')
fig.colorbar(q1, ax=axes[0], label='|B| Magnitude')
axes[0].legend()


q1 = axes[1].quiver(X, Z, E_x, E_y, E_mag, scale=3, scale_units='inches', norm=colors.LogNorm(vmin=1e-20, vmax=1e-17))
axes[1].set_title(f'Electric Field $\mathbf{{E}}$ Quiver (t = {t_snapshot}s, y=1m)')
axes[1].set_xlabel('X Position (m)')
axes[1].set_ylabel('Z Position (m)')
axes[1].set_aspect('equal')
fig.colorbar(q1, ax=axes[1], label='|E| Magnitude')
axes[1].legend()

plt.show()