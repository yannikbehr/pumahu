import plotly.graph_objects as go
import numpy as np
import xarray as xr
import pandas as pd

fn = './data/uks.nc'
xdf = xr.open_dataset(fn)
data = xdf.exp.loc[:, 'q_in', 'val'].values
sigma = xdf.exp.loc[:, 'q_in', 'std'].values
# Convert MW to TJ by multiplying with 86400 s
data *= 0.864
sigma *= 0.864
# Convert to kWh by deviding with 3.6e6 J
data /= 3.6
sigma /= 3.6
cum_mean = np.cumsum(data)
cum_std = np.sqrt(np.cumsum(sigma))
dates = pd.to_datetime(xdf.dates.values)
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=cum_mean, mode='lines',
                         showlegend=False))
fig.add_trace(go.Scatter(name='Upper Bound',
                         x=dates,
                         y=cum_mean + 3*cum_std,
                         mode='lines',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False))
fig.add_trace(go.Scatter(name='Lower Bound',
                         x=dates,
                         y=cum_mean - 3*cum_std,
                         mode='lines',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         fillcolor='rgba(68, 68, 68, 0.3)',
                         fill='tonexty',
                         showlegend=False))
fig.update_yaxes(title='$10^6 kWh$')
fig.update_layout(font_size=22)
fig.write_image('cumulative_heat.png', width=1500, scale=2)