from collections import OrderedDict

import numpy as np

from pumahu.syn_model import (SynModel,
                              setup_realistic,
                              resample,
                              make_sparse)
from pumahu.visualise import trellis_plot
from pumahu.uks import UnscentedKalmanSmoother


xds = SynModel().run(setup_realistic(sinterval=120), addnoise=True)

na = resample(xds)
na = make_sparse(na, ['m_out', 'X'])
Q = OrderedDict(T=5e-1, M=1e-1, X=1e-3, q_in=1e1,
                m_in=1e1, m_out=1e1, h=1e-3, W=1e-3,
                dqi=1e2, dMi=0, dMo=0, dH=0, dW=0)
Q = np.eye(len(Q))*list(Q.values())

uks = UnscentedKalmanSmoother(data=na, Q=Q,
                              initvals={'qi':200., 'm_in': 0,
                                        'm_out':0., 'X':3.})
xds_uks = uks(test=True, smooth=True, alpha=0.35)
fig = trellis_plot(xds_uks, data2=xds,
                   data2_params=['q_in', 'm_in', 'm_out'],
                   filled_error=True)
fig.update_layout(font_size=20, height=1200, showlegend=False)
filename="synthetic_inversion_uks.png"
fig.write_image(filename, scale=2)
