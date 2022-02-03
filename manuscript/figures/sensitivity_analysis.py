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
df3 = select_data(results, Mo=20., Ws=5, Date='2020-01-01', H=6.0, Qi=200)
T = df3.T0.values
qe = df3.Qe.values/0.0864
dT3 = df3.dT.values
dM3 = df3.dM.values
me = df3.Me.values

# Plotting
fig = make_subplots(rows=1, cols=1,
                    subplot_titles=["Qi=200 MW, Ws=5 m/s, H=6 MJ/kg, Mo=20 kt/day, Date=2020-01-01"])

def add_contour_plot(x, y, z):
    return go.Contour(x=x, y=y, z=z,
                      contours=dict(showlabels=True,
                                    labelfont=dict(size=12,
                                                   color='black')),
                      coloraxis="coloraxis")

#fig.add_trace(add_contour_plot(np.unique(T), np.unique(qe), dT3.T), 1, 1)
fig.add_trace(go.Scatter(x=T, y=qe, mode='lines'), 1, 1)

fig.update_xaxes(row=1, col=1, title_text='Lake temperature [\u00b0C]')
fig.update_yaxes(row=1, col=1, title_text='Surface energy loss [MW]')
fig.update_coloraxes(colorscale='RdBu_r', colorbar=dict(len=.5, yanchor='bottom', title_text='dT [\u00b0C/day]'))
fig.update_layout(width=1200, height=800)
fig.update_layout(font_size=22)
fig.write_image("sensitivity_analysis.png")
