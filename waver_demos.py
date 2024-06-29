import napari
import numpy as np
from napari.utils import Colormap
from waver.datasets import run_and_visualize
from waver.simulation import run_multiple_sources

# Define a simulation, 12.8mm, 100um spacing
speed = 343 * np.ones((128, 128))
speed[70:80, 80:90] = 2*343
sim_params = {
    'size': (12.8e-3, 12.8e-3),
    'spacing': 100e-6,
    'duration': 80e-6,
    'min_speed': 343,
    'max_speed': 686,
    'speed': speed,
    'time_step': 50e-9,
    'temporal_downsample': 2,
    'sources': [{
        'location': (6.4e-3, 6.4e-3),
        'period': 5e-6,
        'ncycles': 1,
    }],
    'boundary': 4,
    'edge': 0,
}

wave, speed = run_multiple_sources(**sim_params)

clim = np.percentile(wave, 99)
wave_cmap = Colormap([[0.55, 0, .32, 1], [0, 0, 0, 0], [0.15, 0.4, 0.1, 1]], name='PBlG')
wave_dict = {'colormap': wave_cmap, 'contrast_limits': [-clim, clim], 'name': 'wave',
             'metadata': sim_params, 'interpolation': 'linear'}

speed_cmap = Colormap([[0, 0, 0, 0], [0.7, 0.5, 0, 1]], name='Orange')
speed_dict = {'colormap': speed_cmap, 'opacity': 0.5,
              'name': 'speed', 'metadata': sim_params, 'interpolation': 'linear'}

viewer = napari.Viewer()
viewer.add_image(np.atleast_2d(np.squeeze(wave)), **wave_dict)
viewer.add_image(np.atleast_2d(speed[0, 0]), **speed_dict)

napari.run()