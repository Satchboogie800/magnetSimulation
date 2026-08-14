import numpy as np

def setup_grid(distance, resolution):
    x = np.linspace(-distance, distance, resolution)
    z = np.linspace(-distance, distance, resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.full_like(X, 1)
    return X, Y, Z

def plot_field_snapshot():
    pass

def animated_fields():
    pass