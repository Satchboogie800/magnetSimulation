import numpy as np
def get_magnet_position(t,v=0.5):
    return np.array([v*t, 0., 0.])

def get_magnetic_moment(t, f, m_mag):
    m = np.array([0., 0., m_mag])
    modulation = np.sin(2*f*np.pi*t)
    return m*modulation