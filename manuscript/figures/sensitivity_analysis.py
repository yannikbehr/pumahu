from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pumahu.syn_model import sensitivity_analysis_grid

def select_data(data, **kargs):
    cond = np.ones(data.shape[0], dtype='bool')
    for key in kargs:
        cond_tmp = data[key] == kargs[key]
        cond = cond & cond_tmp
    df =  data.where(cond)
    df.dropna(inplace=True)
    return df

# Run the analysis
results = sensitivity_analysis_grid()

# data selection
results['Date'] = pd.to_datetime([date(1,1,1) + timedelta(days=dt-1) for dt in results['Date']])
df1 = select_data(results, Ws=2.0, T0=25.0, H=6.0, Date='2020-01-01')
qi = df1.Qi.values.reshape((10, 10))
mo = df1.Mo.values.reshape((10, 10))
dT = df1.dT.values.reshape((10, 10))

df2 = select_data(results, Mo=20., T0=25.0, H=6.0, Ws=2.0)
dt = np.datetime_as_string(df2.Date.values.reshape((10, 10)))
qi2 = df2.Qi.values.reshape((10, 10))
dT2 = df2.dT.values.reshape((10, 10))

df3 = select_data(results, Mo=20., Date='2020-01-01', H=6.0, Qi=200)
ws = df3.Ws.values.reshape((10, 10))
T = df3.T0.values.reshape((10, 10))
qe = df3.Qe.values.reshape((10, 10))
dT3 = df3.dT.values.reshape((10, 10))
dM3 = df3.dM.values.reshape((10, 10))

# Plotting
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=["T=25 \u00b0C, H=6 MJ/kg, Ws=2 m/s, Date=2020-01-01",
                                    "T=25 \u00b0C, H=6 MJ/kg, Ws=2 m/s, Mo=20 kt/day",
                                    "Qi=200 MW, H=6 MJ/kg, Mo=20 kt/day, Date=2020-01-01"],
                    vertical_spacing=0.09)

def add_contour_plot(x, y, z):
    return go.Contour(x=x, y=y, z=z,
                      contours=dict(showlabels=True,
                                    labelfont=dict(size=12,
                                                   color='black')),
                      coloraxis="coloraxis")

fig.add_trace(add_contour_plot(np.unique(mo), np.unique(qi), dT), 1, 1)
fig.add_trace(add_contour_plot(np.unique(dt), np.unique(qi2), dT2.T), 2, 1)
fig.add_trace(add_contour_plot(np.unique(ws), np.unique(T), dT3.T), 3, 1)

fig.update_xaxes(row=1, col=1, title_text='Outflow [kt/day]')
fig.update_xaxes(row=2, col=1, title_text='Date')
fig.update_xaxes(row=3, col=1, title_text='Windspeed [m/s]')
fig.update_yaxes(row=1, col=1, title_text='Energy input [MW]')
fig.update_yaxes(row=2, col=1, title_text='Energy input [MW]')
fig.update_yaxes(row=3, col=1, title_text='Lake temperature [\u00b0C]')
fig.update_coloraxes(colorscale='RdBu_r', colorbar=dict(len=.5, yanchor='bottom', title_text='dT [\u00b0C/day]'))
fig.update_layout(width=800, height=1200)
fig.write_image("sensitivity_analysis.png")
