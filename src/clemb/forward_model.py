import numpy as np


def es(T, w, a):
    """
    Energy loss from lake surface in TJ/day. Equations apply for nominal
    surface area of 200000 square metres. Since area is only to a small
    power, this error is negligible.
    """
    # Assumptions:
    # Characteristic length of lake is 500 m
    l = 500
    # Surface temperature about 1 C less than bulk water
    ts = T - 1.0
    # Air temperature is 0.9 C
    t_air = 0.9
    # Saturation vapour pressure at air temperature = 6.5 mBar and
    # humidity around 50%. This is slightly less than 6.577 posted on
    # the Hyperphysics site
    vp_air = 6.5
    # Saturation vapour density at ambient conditions
    vd_air = 0.0022
    # Atmospheric pressure is 800 mBar at Crater Lake
    pa = 800.
    # Specific heat of air 1005 J/kg
    ca = 1005
    # Air density .948 kg/m^3
    da = .948
    # Latent heat of vapourization about 2400 kJ/kg in range 20 - 60 C,
    lh = 2400000.
    # Expressions for H2O properties as function of temperature
    # Saturation vapour Pressure Function from CIMO Guide (WMO, 2008)
    vp = 6.112 * np.exp(17.62 * ts / (243.12 + ts))
    # Saturation vapour Density from Hyperphysics Site
    vd = (.006335 + .0006718 * ts - .000020887 *
          ts * ts + .00000073095 * ts * ts * ts)

    # First term is for radiation, Power(W) = CA(e_wTw^4 - e_aTa^4)
    # Temperatures absolute, a is emissivity, C Stefans Constant
    tk = ts + 273.15
    tl = t_air + 273.15
    er = 5.67E-8 * a * (0.97 * tk**4 - 0.8 * tl**4)

    # Free convection formula from Adams et al(1990)
    # Power (W) = A * factor * delT^1/3 * (es-ea)

    efree = a * 2.2 * (ts - t_air) ** (1 / 3.0) * (vp - vp_air)

    # Forced convection by Satori's Equation
    # Evaporation (kg/s/m2) =
    # (0.00407 * W**0.8 / L**0.2 - 0.01107/L)(es-ea)/pa

    eforced = a * (0.00407 * w**0.8 / l**0.2 - 0.01107 / l) * \
        (vp - vp_air) / pa * lh  # W

    ee = np.sqrt(efree**2 + eforced**2)

    # The ratio of Heat Loss by Convection to that by Evaporation is
    # rhoCp/L * (Tw - Ta)/(qw - qa) #rho is air density .948 kg/m3, Cp
    ratio = da * (ca / lh) * (ts - t_air) / (vd - vd_air)

    # The power calculation is in W. Calculate Energy Loss (TW/day) and
    # evaporative volume loss in kT/day
    ev = 86400 * ee / 2.4e12  # kT/day
    loss = (er + ee * (1 + ratio)) * 86400 * 1.0e-12  # TW/day
    return loss, ev


def fullness(hgt):
    """
    Calculates crater lake volume and area from lake Level
    The formula has large compensating terms, beware!
    Volume is in 1e3 m^3 = kt
    """
    h = hgt.copy()
    a = np.zeros(h.shape)
    vol = np.zeros(h.shape)
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


def esol(ndays, a, month):
    """
    Solar Incident Radiation Based on yearly guess & month.
    """
    pesol = (ndays * a * 0.000015 *
             (1 + 0.5 * np.cos(((month - 1) * 3.14) / 6.0)))
    return pesol


# Second-order Runge-Kutta
def rk2(y, time, dt, derivs, **kargs):
    k0 = dt * derivs(y, time, **kargs)
    k1 = dt * derivs(y + k0, time + dt, **kargs)
    y_next = y + 0.5 * (k0 + k1)
    return y_next


# Fourth-order Runge-Kutta
def rk4(y, time, dt, derivs, **kargs):
    k0 = dt * derivs(y, time, **kargs)
    k1 = dt * derivs(y + 0.5 * k0, time + 0.5 * dt, **kargs)
    k2 = dt * derivs(y + 0.5 * k1, time + 0.5 * dt, **kargs)
    k3 = dt * derivs(y + k2, time + dt, **kargs)
    y_next = y + 1./6.*(k0 + 2 * k1 + 2 * k2 + k3)
    return y_next


def euler(y, time, dt, derivs, **kargs):
    k0 = dt * derivs(y, time, **kargs)
    y_next = y + k0
    return y_next


class model:

    def __init__(self):
        self.steam = None
        self.mevap = None

    def dT(self, state, time, datetime=0., surfacearea=0.,
           volume=0., volcheat=0., meltwater=0., outflow=0.,
           solar=0., enthalpy=6.0, windspeed=4.5):
        cw = 0.0042
        qe, me = es(state[0], windspeed, surfacearea)
        qi = volcheat - meltwater*state[0]*cw
        steam = volcheat / enthalpy
        self.steam = steam
        self.mevap = me
        # energy loss due to outflow
        qo = outflow*state[0]*cw
        g0 = 1./(cw*state[1])*(-qe + solar + qi - qo)
        g1 = meltwater + steam - me - outflow
        # dX/dt = -M_out*(X_t/m_t)
        # X_t is the total amount of a chemical
        # species at time t
        g2 = -outflow*state[2]/state[1]
        return np.array([g0, g1, g2])


def forward_model(y, dt, surfacearea, volume, volcheat,
                  meltwater, outflow, solar, enthalpy, windspeed,
                  method='euler'):
    a = model()
    if method == 'euler':
        y_new = euler(y, 0., dt, a.dT, surfacearea=surfacearea,
                      volume=volume, volcheat=volcheat, meltwater=meltwater,
                      outflow=outflow, solar=solar, enthalpy=enthalpy,
                      windspeed=windspeed)
    elif method == 'rk2':
        y_new = rk2(y, 0., dt, a.dT, surfacearea=surfacearea,
                    volume=volume, volcheat=volcheat, meltwater=meltwater,
                    outflow=outflow, solar=solar, enthalpy=enthalpy,
                    windspeed=windspeed)
    elif method == 'rk4':
        y_new = rk4(y, 0., dt, a.dT, surfacearea=surfacearea,
                    volume=volume, volcheat=volcheat, meltwater=meltwater,
                    outflow=outflow, solar=solar, enthalpy=enthalpy,
                    windspeed=windspeed)
    return (y_new, a.steam, a.mevap)
