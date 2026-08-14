from cmath import nan

import numpy as np
mu_0 = (4*np.pi)*10**(-7)

def compute_retarded_time():
    pass

def compute_e_field():
    pass

# Magnetic field of the dipole
def compute_b_field(grid_point, magnet_position, m, threshold):
    r_vec = grid_point - magnet_position
    r_mag = np.maximum(np.linalg.norm(r_vec, axis=1, keepdims=True), 1e-4)
    m_dot_r = np.sum(m * r_vec, axis=1, keepdims=True)
    term1 = 3 * r_vec * (m_dot_r / (r_mag ** 5))
    term2 = m / (r_mag ** 3)
    B_field = (mu_0 / (4 * np.pi)) * (term1 - term2)
    mask = r_mag.squeeze() < threshold
    B_field[mask] = nan
    return B_field