import numpy as np
import pandas as pd
from scipy.signal import tukey
from scipy.stats import gamma
from clemb.forward_model import forward_model, esol


class SynModel:

    def __init__(self, area=194162):
        self.f = 1/15.
        self.tmax = 30.
        self.a = area

    def outflow(self, level, area=0.2):
        """
        Compute outflow based on bernoulli equation.
        Returns outflow in km^3/day.
        """
        g = 9.81  # m/s^2
        h = level - 45.31
        if h <= 0.:
            return 0.
        v = np.sqrt(2*g*h)
        vol = area * v  # ouflow volume in m^3/s
        vol *= 1e-3 * 86400  # convert to km^3/day
        return vol

    def volume(self, mass, temp):
        """
        Compute volume based on mass and temperature.
        Return volume in km^3
        """
        rho = 1.003 - 0.00033 * temp
        return mass/rho

    def mass(self, volume, temp):
        """
        Return mass based on volume and temperature.
        """
        rho = 1.003 - 0.00033 * temp
        return volume * rho

    def level(self, volume, lakearea):
        """
        Compute lake level based on volume and surface area.
        Return lake level in m.
        """
        return volume * 1e3 / lakearea

    def synth_fullness(self, level):
        print('synth_fullness called')
        vol = self.a * level
        return self.a, vol/1e3

    def run(self, nsteps, q_in, mode='gamma'):
        dates = pd.date_range(start='1/1/2017', end='21/1/2017',
                              periods=nsteps)
        dt = (dates[1] - dates[0])/pd.Timedelta('1D')
        t = np.linspace(0, self.tmax, nsteps)
        if mode == 'sinus':
            qi = np.sin(2.*np.pi*self.f*t + np.deg2rad(15))*q_in/2.+q_in/2.
            qi *= tukey(nsteps, 0.9)
        if mode == 'tukey':
            tail = int(10/dt)
            qi = np.ones(nsteps-tail)*q_in
            qi *= tukey(nsteps-tail, 0.9)
            qi_tmp = np.zeros(nsteps)
            qi_tmp[0:nsteps-tail] = qi
            qi = qi_tmp
        if mode == 'step':
            tail = int(10/dt)
            qi = np.ones(nsteps-tail)*q_in
            qi_tmp = np.zeros(nsteps)
            qi_tmp[0:nsteps-tail] = qi
            qi = qi_tmp
        if mode == 'gamma':
            y = gamma.pdf(t, 3., loc=4)
            y /= y.max()
            y *= q_in
            qi = y

        y = np.zeros((nsteps, 3))
        prm = np.zeros((nsteps, 6))
        T = 15.
        V = 8800
        A = self.a
        Mc = 10.
        H = 6.0
        WS = 4.5
        X = 2.
        M = self.mass(V, T)
        y[0, :] = [T, M, X]
        ll = self.level(V, A)
        Mo = self.outflow(ll)

        for i in range(nsteps-1):
            prm[i, 0] = Mc
            prm[i, 1] = Mo
            prm[i, 2] = ll
            dt = (dates[i+1] - dates[i])/pd.Timedelta('1D')
            solar = esol(1., A, dates[i].month)
            y_new, steam, mevap = forward_model(y[i], dt, A, V, qi[i]*0.0864,
                                                Mc, Mo, solar, H, WS,
                                                method='euler')
            prm[i, 3] = mevap
            prm[i, 4] = solar
            prm[i, 5] = qi[i]
            V = self.volume(y_new[1], y_new[0])
            ll = self.level(V, A)
            Mo = self.outflow(ll)
            y[i+1, :] = y_new[:]
        prm[i+1, 0] = Mc
        prm[i+1, 1] = Mo
        prm[i+1, 2] = ll
        prm[i+1, 3] = mevap
        prm[i+1, 3] = solar

        # Prescripe errors
        factor = 1.
        T_err = factor*0.4*dt
        z_err = factor*0.01*dt
        M_err = factor*2.*dt
        X_err = factor*0.4*dt
        A_err = factor*30.*dt
        V_err = factor*2.0*dt
        Mg_err = factor*50*dt

        # Design dataset
        syn_data = {}
        syn_data['T'] = y[:, 0]
        syn_data['T_err'] = np.ones(nsteps)*T_err
        syn_data['p_mean'] = 1.003 - 0.00033 * syn_data['T']
        syn_data['p_err'] = 0.00033*np.random.normal(scale=T_err, size=nsteps)
        syn_data['a'] = np.ones(nsteps)*A
        syn_data['a_err'] = np.ones(nsteps)*A_err
        syn_data['M'] = y[:, 1]
        syn_data['M_err'] = np.ones(nsteps)*M_err
        syn_data['v'] = self.volume(syn_data['M'], syn_data['T'])
        syn_data['v_err'] = V_err
        syn_data['X'] = y[:, 2]
        syn_data['X_err'] = np.ones(nsteps)*X_err
        syn_data['Mg'] = syn_data['X']/syn_data['M']*1e6
        syn_data['Mg_err'] = Mg_err
        syn_data['z'] = prm[:, 2]
        syn_data['z_err'] = np.ones(nsteps)*z_err
        syn_data['W'] = np.ones(nsteps)*4.5
        syn_data['H'] = np.ones(nsteps)*6.0
        syn_data['dv'] = np.ones(nsteps)*1.0
        return pd.DataFrame(syn_data, index=dates), prm
