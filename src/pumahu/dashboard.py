from dash import Dash, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def dashboard_plot(df_q, df_T):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    mean = df_q.val.values
    mean = np.where(mean < 0., 0., mean)
    color_T = 'rgb(26,166,183)'
    color = 'rgb(255,65,77)'
    dates = df_q.index
    upper_error = mean + 3*df_q['std'].values
    lower_error = mean - 3*df_q['std'].values
    lower_error = np.where(lower_error < 0., 0., lower_error)
    fig.add_trace(go.Scatter(x=df_T.index, y=df_T.val, line=dict(color=color_T, dash='solid'), name='Temperature', showlegend=True), secondary_y=True)
    fig.add_trace(go.Scatter(x=dates, y=mean,line=dict(color=color, dash='solid'), name='Heat input rate', showlegend=True))
    fig.add_trace(go.Scatter(x=dates, y=upper_error, mode='lines', marker=dict(color="#444"), line=dict(width=0),showlegend=False))
    fillcolor = 'rgba(26,166,183,.3)'
    fillcolor = 'rgba(255,65,77, .3)'
    fig.add_trace(go.Scatter(x=dates, y=lower_error, mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False,
                             fillcolor=fillcolor,
                             fill='tonexty'))
    fig.update_yaxes(title_text=u'T [\N{DEGREE SIGN}C]', secondary_y=True, color=color_T)
    fig.update_yaxes(title_text='Q [MW]', secondary_y=False, color=color)
    return fig


def main():
    base_url = "http://volcanolab.gns.cri.nz:11111/v1/data/"
    q_request = "{}?param={}&start={}".format(base_url, 'q_in', '2016-03-04')
    T_request = "{}?param={}&start={}".format(base_url, 'T', '2016-03-04')
    df_q = pd.read_csv(q_request, parse_dates=True, index_col=0)
    df_T = pd.read_csv(T_request, parse_dates=True, index_col=0)

    fig = dashboard_plot(df_q, df_T)
    app = Dash(__name__)
    app.layout = html.Div([dcc.Graph(figure = fig)])
    app.run_server(host='0.0.0.0', debug=True, use_reloader=False)
    

if __name__ == '__main__':
    main()