import os
import xarray as xr
from pumahu.visualise import plot_qin_uks
from pumahu.uks import main as main_uks

fn = 'data/uks.nc'
if not os.path.isfile(fn):
    main_uks(['--rdir', './data', '-s', '2016-03-04',
              '-e', '2021-07-07', '-f'])

xdf = xr.open_dataset('data/uks.nc')
fig = plot_qin_uks(xdf)
fig.update_layout(showlegend=False,
                  font_size=22)
fig.write_image('uks_real_data.png', width=1500)