from cmath import nan
from magnetsimulation.motion import get_magnet_position
from magnetsimulation.motion import get_magnetic_moment
import numpy as np
mu_0 = (4*np.pi)*10**(-7)

def compute_retarded_time():
    pass

def compute_e_field():
    pass

# Magnetic field of the dipole
def compute_static_b_field(grid_point, magnet_position, m, threshold):
    """
    Calculates the B-Field (in 2 Dimensions for now) based on the magnet position and magnetic moment m. Uses a
    threshold to not calculate points to close to the dipole to avoid very large magnitudes because of division by
    r**3!
    :param grid_point: uv, uy from np.meshgrid
    :param magnet_position: Current magnets position
    :param m:  Magnetic moment (m) (2d)
    :param threshold: Min Distance from magnet to calculate field
    :return: Matrix with Bx, By for each gridpoint
    """
    r_vec = grid_point - magnet_position
    r_mag = np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-4)
    m_dot_r = np.sum(m * r_vec, axis=1, keepdims=True)
    term1 = 3 * r_vec * (m_dot_r / (r_mag ** 5))
    term2 = m / (r_mag ** 3)
    B_field = (mu_0 / (4 * np.pi)) * (term1 - term2)
    mask = r_mag.squeeze() < threshold
    B_field[mask] = nan
    return B_field

def compute_b_field(grid_point, magnet_position, m, threshold):
    r_vec = grid_point - magnet_position
    r_mag = np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-4)
    r_hat = r_vec / r_mag
    m_broadcast = m.reshape(([1] * (r_vec.ndim - 1)) + [3])
    m_dot_rhat = np.sum(m_broadcast * r_hat, axis=-1, keepdims=True)
    # m_dot_rhat = np.sum(m_mag * r_hat, axis=1, keepdims=True)
    B = (mu_0 / (4 * np.pi)) * (3 * m_dot_rhat * r_hat - m_broadcast) / (r_mag ** 3)
    return B

def get_field_at_time(t, grid, r_grid, dt=1e-3):

    original_shape = grid.shape
    if len(original_shape) == 3:
        grid_flat = grid.reshape(-1, 3)
        r_grid_flat = r_grid.reshape(-1, 3)
    else:
        grid_flat = grid
        r_grid_flat = r_grid

    pos_c = get_magnet_position(t)
    pos_p = get_magnet_position(t - dt)
    pos_n = get_magnet_position(t + dt)

    m_c = get_magnetic_moment(t, 10, 1000)
    m_p = get_magnetic_moment(t - dt, 10, 1000)
    m_n = get_magnetic_moment(t + dt, 10, 1000)

    B_current = compute_b_field(grid_flat, pos_c, m_c, 1)
    B_prev = compute_b_field(grid_flat, pos_p, m_p, 1)
    B_next = compute_b_field(grid_flat, pos_n, m_n, 1)

    dB_dt = (B_next-B_prev)/(2*dt)
    m_dot = (m_n-m_p) / (2*dt)

    R_vec = r_grid_flat - pos_c
    R_mag = np.maximum(np.linalg.norm(R_vec, axis=1, keepdims=True), 1e-4)
    R_hat = R_vec / R_mag

    m_dot_broadcast = m_dot.reshape(([1] * (R_hat.ndim - 1)) + [3])
    E_current = -(mu_0 / (4 * np.pi)) * np.cross(m_dot_broadcast, R_hat) / (R_mag ** 2)

    # Optional: Reshape outputs back to (25, 25, 3) if your plotting tool expects a meshgrid
    if len(original_shape) == 3:
        B_current = B_current.reshape(original_shape)
        dB_dt = dB_dt.reshape(original_shape)
        E_current = E_current.reshape(original_shape)

    return B_current, dB_dt, E_current