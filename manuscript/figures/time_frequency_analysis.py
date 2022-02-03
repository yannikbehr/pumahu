import pandas as pd
import numpy as np
import os
import xarray as xr

import plotly.graph_objects as go
import pywt

from pumahu.visualise import TrellisPlot
from pumahu.uks import main as main_uks

if False:
    df_modis = pd.read_csv(get_data('data/41561_2021_705_MOESM2_ESM.csv'), usecols=[5, 6, 7, 8],
                            index_col=0, dayfirst=True, parse_dates=True)
    df_modis = df_modis.loc[str(dates[0]):str(dates[-1])]
    
def wavelet_analysis(dates, data, nscales=128):
    sst = data - data.mean()
    dt = (dates[1]-dates[0])/pd.Timedelta('1D')

    # Taken from http://nicolasfauchereau.github.io/climatecode/posts/wavelet-analysis-in-python/
    wavelet = 'cmor1.5-1.0'
    #wavelet = 'cmor2.0-6.0'
    #wavelet = 'mexh'
    scales = np.arange(1, nscales)

    [cfs, frequencies] = pywt.cwt(sst, scales, wavelet, dt)
    power = (abs(cfs)) ** 2

    period = 1. / frequencies
    return period, power

   
fn = 'data/uks.nc'
if not os.path.isfile(fn):
    main_uks(['--rdir', './data', '-s', '2016-03-04',
              '-e', '2022-02-01', '-f'])

xdf = xr.open_dataset(fn)
data = xdf.exp.loc[:, 'q_in', 'val'].values[:]
dates = xdf['dates'].values

tp = TrellisPlot()
ymean, ymin, ymax = tp.get_traces(xdf.exp, 'q_in')
nscales = 256 
period, power = wavelet_analysis(dates, data, nscales)
mean_power = power.mean(axis=1)
std_power = power.std(axis=1)
min_power = mean_power - std_power
max_power = mean_power + std_power
min_power = np.where(min_power < 0., 0., min_power)

fig = go.Figure()
fig.add_trace(go.Contour(
        x = pd.to_datetime(dates),
        y = period,
        z = power,
        contours=dict(
            start=0,
            end=700e3,
            size=100e3,
        ),
        contours_coloring='heatmap',
        colorscale = 'Blues',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y'
    ))

fig.add_trace(go.Scatter(
    x=[pd.to_datetime(dates)[150]],
    y=[10],
    mode="text",
    name="Markers and Text",
    text=["b)"],
    textfont=dict(
        color="white"
    ),
    textposition="bottom center"
))

fig.add_trace(go.Scatter(
        x = pd.to_datetime(dates),
        y = ymean,
        mode='lines',
        xaxis='x',
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ))
fig.add_trace(go.Scatter(x=pd.to_datetime(dates),
                         y=ymin,
                         mode='lines',
                         yaxis='y2',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False))
                
fig.add_trace(go.Scatter(x=pd.to_datetime(dates),
                         y=ymax,
                         mode='lines',
                         yaxis='y2',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False,
                         fillcolor='rgba(68, 68, 68, 0.3)',
                         fill='tonexty'))


fig.add_trace(go.Scatter(
    x=[pd.to_datetime(dates)[150]],
    y=[800],
    mode="text",
    name="Markers and Text",
    text=["a)"],
    yaxis='y2',
    textposition="bottom center"
))

fig.add_trace(go.Scatter(
        y = period,
        x = mean_power,
        mode='lines',
        xaxis = 'x2',
        yaxis = 'y',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
   ))
fig.add_trace(go.Scatter(y=period,
                         x=min_power,
                         mode='lines',
                         xaxis='x2',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False))
                
fig.add_trace(go.Scatter(y=period,
                         x=max_power,
                         mode='lines',
                         xaxis='x2',
                         marker=dict(color="#444"),
                         line=dict(width=0),
                         showlegend=False,
                         fillcolor='rgba(68, 68, 68, 0.3)',
                         fill='tonextx'))

fig.add_trace(go.Scatter(
    x=[100e3],
    y=[10],
    mode="text",
    name="Markers and Text",
    text=["c)"],
    xaxis='x2',
    textposition="bottom center"
))


fig.update_layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.8],
        showgrid = True,
        range = (nscales, 0),
        title = 'Period [days]'
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = True,
        tickvals=[200e3, 400e3, 600e3],
        tickmode='array',
        range = (0, 600e3)
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.8,1],
        showgrid = True,
        tickvals=[200, 600, 1000],
        tickmode='array',
        title = 'Qi [MW]'
    ),
    height = 1000,
    width = 1000,
    hovermode = 'closest',
    showlegend = False
)

fig.update_layout(font_size=22)
fig.write_image('time_frequency_analysis.png', width=1500)