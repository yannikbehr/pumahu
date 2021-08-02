import plotly.graph_objects as go
import numpy as np
from mtspec import mtspec
from scipy import signal
import xarray as xr

fn = 'data/uks.nc'
xdf = xr.open_dataset(fn)
data = xdf.exp.loc[:, 'q_in', 'val'].values
data -= data.mean()
spec, freq, jackknife, _, _ = mtspec(data=data, delta=1.0, time_bandwidth=3.5,
    number_of_tapers=10, nfft=2048, statistics=True)
idx = freq<0.2
freq = np.r_[1./freq[1:], 0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=freq, y=spec, name='Multi taper', mode='lines',
                         showlegend=False))
if False:
    freqs, psd = signal.welch(data, nfft=2048, nperseg=256)
    freqs = np.r_[1./freqs[1:], 0]
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', line_dash='dash',
                             name='Welch'))
fig.add_trace(go.Scatter(name='Upper Bound',
                         x=freq,
                         y=jackknife[:, 1],
                         mode='lines',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False))
fig.add_trace(go.Scatter(name='Lower Bound',
                         x=freq,
                         y=jackknife[:, 0],
                         mode='lines',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         fillcolor='rgba(68, 68, 68, 0.3)',
                         fill='tonexty',
                         showlegend=False))
fig.update_xaxes(type="log", title='Period [days]')
fig.update_layout(font_size=22)
fig.write_image('psd_analysis.png', width=1500, scale=2)