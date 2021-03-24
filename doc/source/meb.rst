Mass and Energy Balance model
-----------------------------

Equations from Hurst et al., 1991
Energy balance for the lake:

.. math::

   \frac{d}{dt}Q = Q_i - Q_e - cTM_o + cT_sM_s


Q = total energy of the lake

T = temperature at any time $t$

c = specific heat of lake water

Q\ :sub:`i` = heat input at the base of the lake

Q\ :sub:`e` = heat loss due to evaporation, radiation, solar heating (gain)

M\ :sub:`o` = total rate of outflow

M\ :sub:`s` = inflow rate from melt

T\ :sub:`s` = temperature of inflow

Assuming 

.. math:: T_s = 0 ^{\circ}C:

.. math::
   
   \frac{d}{dt}Q & = Q_i - Q_e - cTM_o \\
                         & = \frac{d}{dt}[cMT] \\
                         & = cM\frac{dT}{dt} + cT\frac{dM}{dt}

   \Rightarrow \qquad \frac{dT}{dt} = \frac{1}{cM}\left(Q_i - Q_e - cTM_o -cT\frac{dM}{dt}\right) 

M = mass of the water in the lake at time t

Mass balance:

.. math::

   \frac{dM}{dt} = M_i + M_s - M_o - M_e

M\ :sub:`i` = rate at which water or steam is added through the volcanic vent

M\ :sub:`e` = rate of evaporation losses

Ion concentration balance:

.. math::

   \begin{aligned}
   \frac{d}{dt}[MZ] & = Z_i M_i - Z M_o\\
           \frac{dM}{dt} + M_o & = \frac{1}{Z} ( Z_i M_i - M \frac{dZ}{dt} ) 
   \end{aligned}

Z = ion concentration in the lake

Z\ :sub:`i`\ M\ :sub:`i` = rate of addition of ions through the lake bottom

If Z\ :sub:`i`\ M\ :sub:`i` = 0:

.. math:: \frac{dM}{dt} = -M_o - \frac{M}{Z}\frac{dZ}{dt}

In the following are the equations used for surface losses (Q\ :sub:`e` ):

*Long wave radiation loss*:

.. math::

   \dot{E}_{long} = 5.67e^{-8}A[0.97(T - 1 + 273.15)^4 -  0.8(0.9 + 273.15)^4]

A = lake surface area

Free and forced convection

.. math::

   \dot{E}_{evap} = A \sqrt{ \left [ 2.2 (T - 1 - 0.9)^{\frac{1}{3}} (6.112e^{\frac{17.62(T-1)}{243.12(T-1)}} - 6.5) \right ]^2+ \left [(4.07e^{-3} \frac{w^{0.8}}{500^{0.2}} - \frac{1.107e^{-2}}{500}) (\frac{1}{800} (6.112e^{\frac{17.62(T-1)}{243.12(T-1)}} - 6.5) ) 2400000 \right ]^2}

w = wind speed

*Solar heating*:

.. math::

   \dot{E}_{solar} = \Delta t A 1.5e^{-5} \left [ 1 + 0.5 cos (\frac{((m-1)3.14)}{6.0} ) \right ]

m = month of the year

*Heat loss due to evaporation radiation and solar heating*:

.. math::

   Q_e = \dot{E}_{long} + \dot{E}_{evap} \left ( 1+0.948 \frac{1005}{2.4e^6} \frac{T - 1 - 0.9}{6.335e^{-3} + 6.718e^{-4}(T-1) -2.0887e^{-5}(T-1)^2 + 7.3095e^{-7}(T-1)^3 - 2.2e^{-3}} \right ) - \dot{E}_{solar}

.. math:: M_e = \dot{E}_{evap}2.4e^{-12}

.. automodule:: pumahu.Forwardmodel
   :members:

.. autoclass:: pumahu.Forwardmodel
   :members: 


