"""
Code to visualize results from the crater lake energy and mass balance
computations.
"""
from collections import defaultdict
import os
import urllib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
from obspy import read, UTCDateTime, Stream
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.neighbors.kde import KernelDensity

from clemb import get_data


class TrellisPlot:
    def __init__(self):
        self.line_colours = ['rgb(26,166,183)',
                             'rgb(255,65,77)',
                             'rgb(0,45,64)']
        self.error_colours = ['rgba(26,166,183,.3)',
                              'rgba(255,65,77, .3)',
                              'rgba(0,45,64,.3)']
        self.legend_traces = []
        self._ntraces = defaultdict(lambda: -1)

    def get_traces(self, data, key, err_min=1e-3):
        if 'val_std' in data.dims:
            # ignore warnings due to NaNs
            with np.errstate(invalid='ignore'):
                ymean = data.loc[:, key, 'val'].values
                ymean = np.where(ymean < 0., 0., ymean)
                yerr = np.sqrt(data.loc[:, key, 'std'].values)
                ymin = ymean - 3*yerr
                ymin = np.where(ymin < 0., 0., ymin)
                ymax = ymean + 3*yerr
                if np.nanmax(ymax-ymin) < err_min:
                    ymax = ymean + err_min
                    ymin = ymean + err_min
                return [ymean, ymin, ymax]
        else:
            ymean = data.loc[:, key].values
            with np.errstate(invalid='ignore'):
                ymean = np.where(ymean < 0., 0., ymean)
            return [ymean]

    def plot_trace(self, fig, data, key, ylim, ylabel, row,
                   plotvars=[('exp', 'Result'), ('input', 'Input')]):
        """
        :param plotvars: Variables to plot.
        """
        dates = pd.to_datetime(data['dates'].values)
        for source, name in plotvars:
            showlegend = True
            self._ntraces[row] += 1
            try:
                traces = self.get_traces(data[source], key)
            except KeyError:
                msg = "{} not in {}".format(key, data[source].coords)
                print(msg)
                continue
            if name in self.legend_traces:
                showlegend = False
            else:
                self.legend_traces.append(name)

            fig.add_trace(go.Scatter(x=dates, y=traces[0],
                                     line_color=self.line_colours[self._ntraces[row]],
                                     legendgroup=name,
                                     name=name,
                                     showlegend=showlegend), row=row, col=1)
            if len(traces) > 1:
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
            fig.update_yaxes(title_text=ylabel, row=row, col=1)


def trellis_plot(data, data2=None, filename=None):
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

    with plt.style.context('bmh'):
        fig = make_subplots(rows=7, cols=1, shared_xaxes=True,
                            vertical_spacing=.01)
        tp = TrellisPlot()
        tp.plot_trace(fig, data, 'q_in', (-10, 1100), 'Qi [MW]', row=1)
        tp.plot_trace(fig, data, 'm_in', (-1, 50), 'Mi [kt/day]', row=2)
        tp.plot_trace(fig, data, 'm_out', (-1, 50), 'Mo [kt/day]', row=3)
        tp.plot_trace(fig, data, 'T', (13, 21), 'T [C]', row=4)
        tp.plot_trace(fig, data, 'M', (8760, 8800), 'M [kt]', row=5)
        tp.plot_trace(fig, data, 'X', (1., 3), 'X [kt]', row=6)
        tp.plot_trace(fig, data, 'h', (0., 10), 'H [MJ/kg]', row=7)
        if data2 is not None:
            xdf2 = data2.loc[dict(dates=slice(data.dates[0],
                                              data.dates[-1]))]
            tp.plot_trace(fig, xdf2, 'q_in', (-10, 1100), 'Qi [MW]', row=1,
                          plotvars=[('exp', 'Benchmark')])
            tp.plot_trace(fig, xdf2, 'm_in', (-1, 50), 'Mi [kt/day]', row=2,
                          plotvars=[('exp', 'Benchmark')])
            tp.plot_trace(fig, xdf2, 'm_out', (-1, 50), 'Mo [kt/day]', row=3,
                          plotvars=[('exp', 'Benchmark')])

    fig.update_layout(template='ggplot2',
                      height=1000,
                      width=1200)

    fig.update_xaxes(showticklabels=False, ticks='inside')
    fig.update_xaxes(showticklabels=True, row=7)
    fig.update_yaxes(row=7, col=1, title=dict(standoff=10))
    fig.update_yaxes(row=6, col=1, title=dict(standoff=40))
    fig.update_yaxes(row=5, col=1, title=dict(standoff=20))
    fig.update_yaxes(row=4, col=1, title=dict(standoff=38))
    fig.update_yaxes(row=3, col=1, title=dict(standoff=38))
    fig.update_yaxes(row=2, col=1, title=dict(standoff=38))
    fig.update_yaxes(row=1, col=1, title=dict(standoff=20))
    if filename is not None:
        fig.write_image(file=filename)
    return fig


def density_plot(ax, data, prm, prm_lim=(0, 1600, 1000),  bw=60., mode='kde'):
    nsteps = data.dims['dates']
    X_plot = np.linspace(*prm_lim)
    if mode == 'kde':
        m = []
        for i in range(nsteps):
            y = data.ns_samples.loc[dict(parameters=prm)][dict(dates=i)].data
            idx = np.isnan(y)
            if np.all(idx):
                m.append(np.zeros(X_plot.size))
                continue
            _kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(y[~idx].reshape(-1, 1))
            log_dens = _kde.score_samples(X_plot[:, np.newaxis])
            m.append(np.exp(log_dens))
        m = np.array(m)
        ax.contourf(np.arange(nsteps), X_plot, m.T, 30, cmap=plt.cm.get_cmap('RdBu_r'))
    if mode == 'scatter':
        datet = data.dates
        nresample = data.ns_samples.shape[1]
        for k in range(nsteps):
            y = data.ns_samples.loc[dict(parameters=prm)][dict(dates=k)].data
            c = data.p_samples.loc[dict(p_parameters='lh')][dict(dates=k)].data
            ax.scatter([datet[k].data]*nresample, y, s=2, c=c,
                       cmap=plt.cm.get_cmap('RdBu_r'), alpha=0.3)
    return


def adjust_labels(ax, dates, rotation=30):
    new_labels = []
    new_ticks = []
    for _xt in ax.get_xticks():
        try:
            dt = dates[int(_xt)].astype('datetime64[us]').min()
            new_labels.append((pd.to_datetime(str(dt)).strftime("%Y-%m-%d")))
            new_ticks.append(_xt)
        except IndexError:
            continue
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(new_labels, rotation=rotation, horizontalalignment='right')


def mcmc_heat_input(data, filename=None):
    fig = plt.figure(figsize=(16, 5))
    ax1 = fig.add_axes([0.01, 0.1, 0.8, 0.9])
    ax2 = fig.add_axes([0.82, 0.1, 0.15, 0.9])
    axs = [ax1, ax2]

    density_plot(axs[0], data, 'q_in', prm_lim=(0, 1100, 1000), mode='kde')
    exp_q_in = data.exp.loc[dict(parameters='q_in', val_std='val')]
    axs[0].plot(exp_q_in, 'k')
    adjust_labels(axs[0], data.dates.data)
    axs[0].set_ylabel('Energy input rate [MW]')
    axs[0].set_xlim(0, data.dims['dates']-1)
    axs[0].set_ylim(0, 1100)

    X_plot = np.linspace(0, 1600, 1000)[:, np.newaxis]
    for i in [-2, -3, -4, -5]:
        y = data.ns_samples.loc[dict(parameters='q_in')][dict(dates=i)].data
        idx = np.isnan(y)
        kde = KernelDensity(kernel='gaussian', bandwidth=60.).fit(y[~idx].reshape(-1, 1))
        log_dens = kde.score_samples(X_plot)
        Y = np.exp(log_dens)
        date_str = pd.to_datetime(data['dates'].data[i]).strftime("%Y-%m-%d")
        if i == -2:
            axs[1].plot(Y, X_plot[:, 0], linewidth=5., label=date_str)
        else:
            axs[1].plot(Y, X_plot[:, 0], label=date_str)

    axs[1].legend()
    axs[1].yaxis.tick_right()
    axs[1].set_xticks([])
    axs[1].set_ylim(0, 1100)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig


def get_rsam_data(filename, outdir):
    baseurl = 'http://vulkan.gns.cri.nz:9090/MAVZ.NZ/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    fout = os.path.join(outdir, filename)
    if not os.path.isfile(fout):
        hr = urllib.request.urlopen(os.path.join(baseurl, filename))
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
                 '2021.MAVZ.10-HHZ.NZ.bp_1.00-4.00.rsam']
    for _fn in rsamfiles:
        outdir = os.path.join(os.environ['HOME'], '.cache', 'clemb_mcmc')
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
