import numpy as np

def setup_grid(distance, resolution):
    x = np.linspace(-distance, distance, resolution)
    y = np.linspace(-distance, distance, resolution)
    xc, yv = np.meshgrid(x, y)
    return xc, yv

def plot_field_snapshot():
    pass

def animated_fields():
    pass