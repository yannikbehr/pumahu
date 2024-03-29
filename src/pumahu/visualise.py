"""
Code to visualize results from the crater lake energy and mass balance
computations.
"""
from collections import defaultdict
import os
import urllib
import ssl

import matplotlib.pyplot as plt
import numpy as np
from obspy import read, UTCDateTime, Stream
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TrellisPlot:
    def __init__(self):
        self.line_colours = ['rgb(26,166,183)',
                             'rgb(255,65,77)',
                             'rgb(0,45,64)']
        self.line_dash = ['solid', 'dash', 'dash'] 
        self.error_colours = ['rgba(26,166,183,.3)',
                              'rgba(255,65,77, .3)',
                              'rgba(0,45,64,.3)']
        self.legend_traces = []
        self._ntraces = defaultdict(lambda: -1)

    def get_traces(self, data, key, err_min=1e-3,
                   nanthresh=0.4, dropzeros=True):
        """
        Extract traces
        
        Parameters:
        -----------
        data : :class:`xarray.DataArray`
               Data structure that contains the values
               and error estimates.
        key : str
              Dimension name of the data to extract.
        err_min : float
                  Minimum error to plot in case max
                  and min traces are too close.
        nanthresh : float
                    Interpolate if the number of NaNs is
                    less than nanthresh.
        """
        if 'val_std' in data.dims:
            # ignore warnings due to NaNs
            ymean = data.loc[:, key, 'val'].values
            if dropzeros:
                ymean = np.where(ymean < 0., 0., ymean)
            yerr = data.loc[:, key, 'std'].values
            # Interpolate over negative values
            idx = np.where(yerr<0)[0]
            yerr[idx] = np.nan
            # calculate the percentage of NaN values
            nanperc = float(np.sum(np.isnan(ymean))/data.shape[0])
            if nanperc < nanthresh:
                yerr = pd.Series(yerr).interpolate().values
                ymean = pd.Series(ymean).interpolate().values
            ymin = ymean - 3*yerr
            if dropzeros:
                ymin = np.where(ymin < 0., 0., ymin)
            ymax = ymean + 3*yerr
            if np.nanmax(ymax-ymin) < err_min:
                ymax = ymean + err_min
                ymin = ymean + err_min
            return [ymean, ymin, ymax]
        else:
            ymean = data.loc[:, key].values
            with np.errstate(invalid='ignore'):
                if dropzeros:
                    ymean = np.where(ymean < 0., 0., ymean)
            return [ymean]

    def plot_trace(self, fig, data, key, ylabel, row, filled_error=True,
                   plotvars=[('input', 'Input'),('exp', 'Result')],
                   dropzeros=True, showerror=True):
        """
        :param plotvars: Variables to plot.
        """
        dates = pd.to_datetime(data['dates'].values)
        for source, name in plotvars:
            showlegend = True
            self._ntraces[row] += 1
            if source in data.variables:
                try:
                    traces = self.get_traces(data[source], key,
                                             dropzeros=dropzeros)
                except KeyError:
                    msg = "{} not in {}".format(key, data[source].coords)
                    print(msg)
                    continue
                if name in self.legend_traces:
                    showlegend = False
                else:
                    self.legend_traces.append(name)
                lc = self.line_colours[self._ntraces[row]]
                dash = self.line_dash[self._ntraces[row]]
                if not np.any(np.isnan(traces[0])):
                    fig.add_trace(go.Scatter(x=dates, y=traces[0],
                                             line=dict(color=lc,
                                                       dash=dash),
                                             legendgroup=name,
                                             name=name,
                                             showlegend=showlegend),
                                  row=row, col=1)
                if len(traces) > 1 and showerror:
                    if np.any(np.isnan(traces[0])):
                        error_y = (traces[2] - traces[1])/2.
                        fig.add_trace(go.Scatter(x=dates, y=traces[0],
                                                 mode='markers',
                                                 legendgroup=name,
                                                 name=name,
                                                 showlegend=showlegend,
                                                 error_y=dict(type='data',
                                                              color=lc,
                                                              array=error_y,
                                                              visible=True),
                                                 marker=dict(color=lc, size=5)),
                                      row=row, col=1)
                    elif filled_error:
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=traces[1],
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            showlegend=False), row=row, col=1)
                        fillcolor = self.error_colours[self._ntraces[row]]
                        fig.add_trace(go.Scatter(x=dates,
                                                 y=traces[2],
                                                 mode='lines',
                                                 marker=dict(color="#444"),
                                                 line=dict(width=0),
                                                 showlegend=False,
                                                 fillcolor=fillcolor,
                                                 fill='tonexty'), row=row, col=1)
                    else:
                        for tr in traces[1:]:
                            fig.add_trace(go.Scatter(x=dates, y=tr,
                                                     mode='lines',
                                                     line = dict(color=lc,
                                                                 width=.5,
                                                                 dash='dash'),
                                                      showlegend=False),
                                          row=row, col=1)
                fig.update_yaxes(title_text=ylabel, row=row, col=1)


def trellis_plot(data, data2=None, data2_params=None,
                 filename=None, data2_showerror=True,
                 **kwds):
    """
    A trellis plot for inversion results.

    :param data: Inversion result.
    :type data: :class:`xarray.Dataset`
    :param data2: Results from a second inversion to compare
                 to the first one.
    :type data2: :class:`xarray.Dataset`
    :param filename: File to save the plot to.
    :type filename: str
    """

    params = list(data.parameters.values)
    nparams = len(params)
    labels = dict(q_in='Qi [MW]', m_in='Mi [kt/day]',
                  m_out='Mo [kt/day]', T='T [C]',W='W [m/s]',
                  M='M [kt]', X='X [kt]', h='H [MJ/kg]')
    rowdict = {}
    with plt.style.context('bmh'):
        fig = make_subplots(rows=nparams, cols=1, shared_xaxes=True,
                            vertical_spacing=.01)
        tp = TrellisPlot()
        for i, p in enumerate(params):
            # Don't show q_in as a data input
            plotvars=[('input', 'Input'),('exp', 'Result')]
            if p == 'q_in':
                plotvars=[('NaN', 'NaN'),('exp', 'Result')]
            tp.plot_trace(fig, data, p, labels.get(p, p),
                          plotvars=plotvars, row=i+1, **kwds)
            rowdict[p] = i+1

        if data2 is not None:
            xdf2 = data2.loc[dict(dates=slice(data.dates[0],
                                              data.dates[-1]))]
            if data2_params is None:
                data2_params = params
            for i, p in enumerate(data2_params):
                tp.plot_trace(fig, xdf2, p, labels.get(p, p), row=rowdict[p],
                              plotvars=[('exp', 'Benchmark')],
                              showerror=data2_showerror, **kwds)

    fig.update_layout(template='ggplot2',
                      height=1000,
                      width=1200)

    fig.update_xaxes(showticklabels=False, ticks='inside')
    fig.update_xaxes(showticklabels=True, row=nparams)
    if filename is not None:
        fig.write_image(file=filename, width=1500)
    return fig


def plot_qin_uks(data_uks, data_mcmc=None, data2y=None, filename=None,
                 annotations=False, showlegend=False):
    """
    Plot the MAP of the heat input rate for the 
    Unscented Kalman Smoother solution.
    Parameters:
    -----------
    data_uks: :class:`xarray.DataArray`
              The MAP values from the UKS solution.
    data_mcmc: :class:`xarray.DataArray`
               The MAP values from the MCMC solution
    data2y: (list-like, list-like, str)
            Data to plot on the secondary y-axes in order
            to compare it to the heat input rate. The last
            entry in the tuple defines the legend name.
    filename : str
               Filename to save output to.
    """
    tp = TrellisPlot()
    with plt.style.context('bmh'):
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        tp.plot_trace(fig, data_uks, 'q_in', 'Qi [MW]', 1, filled_error=True,
                      plotvars=[('exp', 'UKS')])
        if data_mcmc is not None:
            tp.plot_trace(fig, data_mcmc, 'q_in', 'Qi [MW]', 1, filled_error=True,
                          plotvars=[('exp', 'MCMC')])
        if data2y is not None:
            x, y, name = data2y
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name),
                          secondary_y=True)
            fig.update_yaxes(title_text=name, secondary_y=True)
        if annotations:
            title = "Heat input rate Ruapehu Crater Lake (Te Wai a-moe)"
            fig.update_layout(title=dict(text=title, x=0.3, y=0.85,
                                         xanchor='center', yanchor='top'))
            ts = pd.to_datetime(data_uks['dates'][-1].values).strftime("%Y-%m-%d")
            latest_val = data_uks.exp.isel(dict(dates=-1, parameters=3, val_std=0)).values
            latest_std = np.sqrt(data_uks.exp.isel(dict(dates=-1, parameters=3, val_std=1)).values)
            text="Latest ({}): {:.0f} +- {:.0f} MW".format(ts, latest_val, latest_std, ts)
            fig.add_annotation(text=text, xref="paper", yref="paper",
                               x=0.3, y=-0.2, showarrow=False)
            fig.update_layout(title=dict(text=title, x=0.3, y=0.85,
                                         xanchor='center', yanchor='top'))
        fig.update_layout(showlegend=showlegend)
    if filename is not None:
        fig.write_image(filename, width=1500)
    return fig


def get_rsam_data(filename, outdir):
    # don't verify the certificate
    myssl = ssl.create_default_context();
    myssl.check_hostname=False
    myssl.verify_mode=ssl.CERT_NONE

    baseurl = 'https://volcanolab.gns.cri.nz:8080/rsam/MAVZ.NZ/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fout = os.path.join(outdir, filename)
    hr = urllib.request.urlopen(os.path.join(baseurl, filename),
                                context=myssl)
    with open(fout, 'wb') as fh:
        fh.write(hr.read())
    return fout


def heat_vs_rsam(data, filename=None):
    dates = data.dates.data
    t1 = UTCDateTime(pd.to_datetime(dates[0]))
    t2 = UTCDateTime(pd.to_datetime(dates[-1]))
    st = Stream()
    rsamfiles = ['2019.MAVZ.10-HHZ.NZ.bp_1.00-4.00.rsam',
                 '2020.MAVZ.10-HHZ.NZ.bp_1.00-4.00.rsam',
                 '2021.MAVZ.10-HHZ.NZ.bp_1.00-4.00.rsam',
                 '2022.MAVZ.10-HHZ.NZ.bp_1.00-4.00.rsam']
    for _fn in rsamfiles:
        outdir = os.path.join(os.environ['HOME'], '.cache', 'pumahu')
        st_tmp = read(get_rsam_data(_fn, outdir))
        tr = st_tmp[0]
        tr.stats.delta = 86400.0
        tr.stats.sampling_rate = 1./tr.stats.delta
        st += tr

    st.merge(fill_value='interpolate')
    st1 = st.trim(t1, t2)
    tr = st1[0]

    exp_q_in = data.exp.loc[dict(parameters='q_in', val_std='val')]
    std_q_in = np.sqrt(data.exp.loc[dict(parameters='q_in', val_std='std')])
    min_q = exp_q_in - std_q_in
    max_q = exp_q_in + std_q_in
    min_q = np.where(min_q < 0, 0., min_q)

    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_axes([0.01, 0.1, 0.8, 0.9])
    ax.plot(dates, exp_q_in, 'k')
    ax.fill_between(dates, min_q, max_q, color='k', alpha=0.2)
    ax.set_ylabel('Energy input rate [MW]')
    ax1 = ax.twinx()
    ax1.plot(tr.times('matplotlib'), tr.data, color='#1f77b4')
    ax1.set_ylabel('RSAM', color='#1f77b4')
    ax1.spines['right'].set_color('#1f77b4')
    ax1.xaxis.label.set_color('#1f77b4')
    ax1.tick_params(axis='y', colors='#1f77b4')
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig
