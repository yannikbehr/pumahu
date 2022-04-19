from tkinter import W
import pandas as pd
import numpy as np
import os
import xarray as xr

import plotly.graph_objects as go
import pywt

from pumahu.visualise import TrellisPlot
from pumahu.uks import main as main_uks
import pycwt
from pycwt.helpers import find


if False:
    df_modis = pd.read_csv(get_data('data/41561_2021_705_MOESM2_ESM.csv'), usecols=[5, 6, 7, 8],
                            index_col=0, dayfirst=True, parse_dates=True)
    df_modis = df_modis.loc[str(dates[0]):str(dates[-1])]
    
def wavelet_analysis(dates, data, wavelet_pkg='pycwt'):
    if wavelet_pkg == 'pywt':
        nscales = 256
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
    elif wavelet_pkg == 'pycwt':
        dt = (np.diff(pd.to_datetime(dates))/pd.Timedelta('1D'))[0]
        N = data.size
        t = np.arange(0, N) * dt
        p = np.polyfit(t, data, 1)
        data_notrend = data - np.polyval(p, t)
        std = data_notrend.std()  # Standard deviation
        var = std ** 2  # Variance
        data_norm = data_notrend / std  # Normalized dataset
        mother = pycwt.Morlet(6)
        s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
        dj = 1 / 12  # Twelve sub-octaves per octaves
        J = 8 / dj  # Seven powers of two with dj sub-octaves
        alpha, _, _ = pycwt.ar1(data)  # Lag-1 autocorrelation for red noise
        wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(data_norm, dt, dj, s0, J,
                                                      mother)
        iwave = pycwt.icwt(wave, scales, dt, dj, mother) * std
        power = (np.abs(wave)) ** 2
        fft_power = np.abs(fft) ** 2
        period = 1 / freqs
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
period, power = wavelet_analysis(dates, data)
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
        ncontours=7,
        contours_coloring='heatmap',
        colorscale = 'Blues',
        reversescale = True,
        xaxis = 'x',
        yaxis = 'y',
        line_smoothing=1.3,
        connectgaps=True
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
        ),
        line_color = 'rgb(255,65,77)'
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
                         fillcolor='rgba(255, 65, 77, 0.3)',
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
    x=[30],
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
        domain = [0,0.7],
        showgrid = True,
        range = np.array([nscales, 0]),
        title = 'Period [days]'
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = True,
        tickvals=[20, 40, 60],
        tickmode='array',
        range = (0, 70)
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.7,1],
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