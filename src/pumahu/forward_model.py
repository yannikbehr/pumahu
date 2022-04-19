from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import brentq


class Forwardmodel:

    def __init__(self, method='euler', level2volume=None, mass2area=None):
        """
        Physical model of the crater lake energy and mass balance.

        Besides the physical model this class also provides different methods
        to integrate the ODEs using either Eulers method, 2nd order Runge-Kutta
        or 4th order Runge-Kutta.

        Parameters
        ----------
        method : string
                 The integration method. Can be either 'euler'
                 (Euler's method), 'rk2' (2nd order Runge-Kutta),
                 or 'rk4' (4th order Runge-Kutta)

        level2volume : function
                       A function to compute lake volume and surface area from
                       the lake level. Takes the lake level as input and return
                       lake surface area and volume (in that order).

        mass2area : function
                    A function to compute lake surface area from the lake mass
                    and temperature. Takes the mass and temperature as input
                    and return surface area and volume (in that order).
        """

        self.steam = None
        self.evap = None
        if method == 'euler':
            self.int_method = self.euler
        elif method == 'rk2':
            self.int_method = self.rk2
        elif method == 'rk4':
            self.int_method = self.rk4
        else:
            msg = 'Please choose one of the following'
            msg += 'integration methods: euler, rk2, rk4.'
            raise ValueError(msg)

        if level2volume is None:
            self.level2volume = self.fullness
        else:
            self.level2volume = level2volume

        if mass2area is None:
            self.mass2area = self.inverse_fullness
        else:
            self.mass2area = mass2area

    def get_evap(self):
        return self.evap

    def get_steam(self):
        return self.steam

    def surface_loss(self, T, w, a, T_air=0.9, l=500., vp_air=6.5,
                     vd_air=0.0022, pa=800., ca=1005., da=0.948,
                     lh=2400000.):
        """
        Energy and mass loss rate from lake surface.

        Equations apply for a nominal surface area of 200000 square metres.
        Since the area is only to a small power, this error is negligible.

        Parameters
        ----------
        T : float or array_like
            Bulk water temperature [degC]
        w : float or array_like
            Wind speed [m/s]
        a : float or array_like
            Lake surface area [m^2]
        l : float, optional
            Characteristic length of lake (i.e. fetch)
        T_air : float, optional
                Air temperature at the lake surface [degC]
        vp_air : float, optional
                 Saturation vapour pressure at air temperature = 6.5 mBar and
                 humidity around 50%. This is slightly less than 6.577 posted
                 on the Hyperphysics site
        vd_air : float, optional
                 Saturation vapour density at ambient conditions
        pa : float, optional
             Atmospheric pressure [mBar]
        ca : float, optional
             Specific heat of air [J/kg]
        da : float, optional
             Air density [kg/m^3]
        lh : float, optional
             Latent heat of vapourization (about 2400 kJ/kg in
             range 20 - 60 degC)

        Returns
        -------
        loss : float or array
               Energy loss due to combined surface losses [TW/day]
        ev : float or array
             Evaporated volume [kT/day]
        """

        # Surface temperature about 1 C less than bulk water
        ts = T - 1.0
        # Expressions for H2O properties as function of temperature
        # Saturation vapour Pressure Function from CIMO Guide (WMO, 2008)
        vp = 6.112 * np.exp(17.62 * ts / (243.12 + ts))
        # Saturation vapour Density from Hyperphysics Site
        vd = (.006335 + .0006718 * ts - .000020887 *
              ts * ts + .00000073095 * ts * ts * ts)

        # First term is for radiation, Power(W) = CA(e_wTw^4 - e_aTa^4)
        # Temperatures absolute, a is emissivity, C Stefans Constant
        tk = ts + 273.15
        tl = T_air + 273.15
        er = 5.67E-8 * a * (0.97 * tk**4 - 0.8 * tl**4)

        # Free convection formula from Adams et al(1990)
        # Power (W) = A * factor * delT^1/3 * (es-ea)

        efree = a * 2.2 * (ts - T_air) ** (1 / 3.0) * (vp - vp_air)

        # Forced convection by Satori's Equation
        # Evaporation (kg/s/m2) =
        # (0.00407 * W**0.8 / L**0.2 - 0.01107/L)(es-ea)/pa

        w = np.where(w < 0, 0, w) # negative wind speeds break the model
        eforced = a * (0.00407 * w**0.8 / l**0.2 - 0.01107 / l) * \
            (vp - vp_air) / pa * lh  # W

        ee = np.sqrt(efree**2 + eforced**2)

        # The ratio of Heat Loss by Convection to that by Evaporation is
        # rhoCp/L * (Tw - Ta)/(qw - qa) #rho is air density .948 kg/m3, Cp
        ratio = da * (ca / lh) * (ts - T_air) / (vd - vd_air)

        # The power calculation is in W. Calculate Energy Loss (TW/day) and
        # evaporative volume loss in kT/day
        ev = 86400 * ee / 2.4e12  # kT/day
        loss = (er + ee * (1 + ratio)) * 86400 * 1.0e-12  # TW/day
        return loss, ev

    def fullness(self, hgt):
        """
        Crater lake volume and area from lake Level

        The formula has large compensating terms, beware!

        Parameters
        ----------
        hgt : float or array_like
              Elevation of lake surface in m a.s.l.

        Returns
        -------
        a : float or array_like
            Lake surface area [m^2]
        vol : float or array_like
              Lake volume [kt]

        """

        h = hgt.copy()
        a = np.zeros(h.shape)*np.nan
        vol = np.zeros(h.shape)*np.nan
        idx1 = np.where(h < 2400.)
        idx2 = np.where(h >= 2400.)
        h[idx1] = 2529.4
        vol[idx1] = (4.747475 * np.power(h[idx1], 3) -
                     34533.8 * np.power(h[idx1], 2) + 83773360. *
                     h[idx1] - 67772125000.) / 1000.
        a[idx1] = 193400
        # Calculate from absolute level
        vol[idx2] = 4.747475 * np.power(h[idx2], 3) - \
            34533.8 * np.power(h[idx2], 2) + 83773360. * h[idx2] - \
            67772125000.
        h[idx2] += 1.0
        v1 = 4.747475 * np.power(h[idx2], 3) - \
            34533.8 * np.power(h[idx2], 2) + 83773360. * h[idx2] - \
            67772125000.
        a[idx2] = v1 - vol[idx2]
        vol[idx2] /= 1000.
        return a, vol

    def inverse_fullness(self, mass, temperature):
        """
        Infer lake level, volume, and lake area from mass and temperature.

        Parameters
        ----------
        mass : float
               Mass of the lake [1e6kg]
        temperature : float
               Water temperature of the lake [degC]

        Returns
        -------
        area : float
               Lake surface area [m^2]
        volume : float
                 Volume of the lake [1e3m^3]
        """
        p = 1.003 - 0.00033 * temperature
        v = mass / p

        def f(h, v):
            a_calc, v_calc = self.fullness(np.array([h]))
            return (v_calc - v)
        z = brentq(f, 2400, 2600, args=v)
        return (*self.fullness(np.array([z])), z)

    def esol(self, a, dtime):
        """
        Solar Incident Radiation per day based on yearly guess & month.

        This is a rough estimate of the short wavelength
        solar incident radiation.

        Parameters
        ----------
        a : float
            Lake surface [m^2]
        month : int
                Month of the year [1-12]

        Returns
        -------
        pesol : float or array_like
                Total energy gain due to short wavelength radiation []

        """
        month = dtime.month
        pesol = (a * 0.000015 * (1 + 0.5 * np.cos(((month - 1) * 3.14) / 6.0)))
        return pesol

    def rk2(self, y, time, dt, **kargs):
        """
        ODE integration with second-order Runge-Kutta
        """
        k0 = dt * self.derivs(y, time, **kargs)
        k1 = dt * self.derivs(y + 0.5 * k0, time + timedelta(days=0.5*dt),
                              **kargs)
        y_next = y + k1
        return y_next

    def rk4(self, y, time, dt, **kargs):
        """
        ODE integration with fourth-order Runge-Kutta
        """
        k0 = dt * self.derivs(y, time, **kargs)
        k1 = dt * self.derivs(y + 0.5 * k0, time + timedelta(days=0.5*dt),
                              **kargs)
        k2 = dt * self.derivs(y + 0.5 * k1, time + timedelta(days=0.5*dt),
                              **kargs)
        k3 = dt * self.derivs(y + k2, time + timedelta(days=dt),
                              **kargs)
        y_next = y + 1./6.*(k0 + 2 * k1 + 2 * k2 + k3)
        return y_next

    def euler(self, y, time, dt, **kargs):
        """
        ODE integration with Euler's rule
        """
        k0 = dt * self.derivs(y, time, **kargs)
        y_next = y + k0
        return y_next

    def derivs(self, state, time, dp=None):
        """
        ODEs describing the change in temperature, mass, and ion concentration
        per day.

        Parameters
        ----------
        state : array_like 
                The state array has to have the following order
                [temperature (C), lake mass (kt), total ion amount (kt),
                 volcanic heat input rate (TJ/day), mass inflow rate (kt/day),
                 mass outflow rate (kt/day), enthalpy (MJ/kg),
                 wind speed (m/s)]

        time : datetime.datetime 
               Time at which the derivatives are computed. This is
               only important for the solor incident radiation.

        dp : array_like 
             The gradient of the model parameters in the following
             [volcanic heat input rate, mass inflow rate,
              mass outflow rate, enthalpy, wind speed]


        Returns:
        --------

        :class:`numpy.ndarray`
                derivatives for:

                [temperature (C/day), lake mass (kt/day), total ion amount (kt/day),
                 volcanic heat input rate (TJ/day/day), mass inflow rate (kt/day/day),
                 mass outflow rate (kt/day/day), enthalpy (MJ/kg/day),
                 wind speed (m/s/day)]

        """
        cw = 0.0042
        a, v, z = self.mass2area(state[1], state[0])
        qe, me = self.surface_loss(state[0], state[7], float(a))
        qi = state[3] - state[4] * state[0] * cw
        steam = state[3] / state[6]
        solar = self.esol(float(a), time)
        self.steam = steam
        self.evap = (qe, me)
        # energy loss due to outflow
        qo = state[5]*state[0]*cw
        g1 = state[4] + steam - me - state[5]
        g0 = 1./(cw*state[1])*(-qe + solar + qi - qo) - state[0]/state[1]*g1
        # dX/dt = -M_out*(X_t/m_t)
        # X_t is the total amount of a chemical
        # species at time t
        g2 = -state[5]*state[2]/state[1]
        return np.r_[np.array([g0, g1, g2]), dp]

    def integrate(self, state, time, dt, dp):
        """
        Do the forward integration using the integration method
        chosen at class initialisation. It accepts the same parameters
        as the `Forwardmodel.derivs` method.
        """
        return self.int_method(state, time, dp=dp, dt=dt)
