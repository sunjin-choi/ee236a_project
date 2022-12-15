
import os 
import sys
import inspect

currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pyLLE
import numpy as np
from scipy import constants as cts
import pandas as pd
import pickle as pkl
import plotly.graph_objs as go
import plotly.graph_objs
import plotly.io as pio
import time
pio.renderers.default = "notebook"
pio.templates.default = "seaborn"


res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        γ=3.2, 
        dispfile="./RW1000_H430.csv"
)

sim = dict(
    Pin=[160e-3], 
    f_pmp=[283e12],
    φ_pmp=[0], 
    δω=[None], 
    Tscan=0.7e6,
    μ_sim=[-220, 220],
    δω_init= 1e9 * 2 * np.pi,
    δω_end= -3.5e9 * 2 * np.pi,
)



solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)

fig = solver.Analyze(plot=True)
# fig.update_yaxes(range = [-50, 50], title = 'D<sub>int</sub> (GHz)')
fig.show()

solver.disp
solver.sim
solver.res

solver.Setup(verbose = False)
solver.SolveTemporal()
solver.RetrieveData()

solver.sol

fig = solver.PlotCombSpectra(4000, do_matplotlib=True)
fig.show()

# fig = go.Figure()
#
# tr = go.Scatter(y=solver.sol.Pcomb * 1e3)
#
# fig.add_trace(tr)
# fig.update_layout(
#     xaxis_title="LLE sub-sampled step", yaxis_title="Intra-cavity Power (mW)"
# )
# fig.add_annotation(x=4300, y=4.9, ax=0, ay=-50, text="Single Soliton Step")
# fig.add_annotation(x=3151, y=7.6, ax=0, ay=-50, text="Two Solitons Step")
# fig.add_annotation(x=1931, y=16, ax=0, ay=80, text="Modulation Instability")
# fig.add_annotation(x=971, y=6, ax=-80, ay=-80, text="Primary Comb")
# fig.show()
