
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

df = pd.read_csv('./DKS_init.csv')
df["DKS_init"] = df.DKS_init.apply(lambda val: complex(val.strip('()')))
δω = df.det.values[0] * 2*np.pi
DKS_init = df.DKS_init.values

res = dict(
        R=23e-6, 
        Qi=1e6, 
        Qc=1e6, 
        γ=3.2, 
        dispfile="./RW1000_H430.csv"
)

sim = dict(
    Pin=[160e-3], # need to be the same length than fmp
    f_pmp=[283e12], # if only one pump f_pmp=[283e12]
    φ_pmp=[0], # need to be the same length than fmp
    δω=[None], # None defined which pump to sweep
    Tscan=0.2e6,
    μ_sim=[-220, 220],
    δω_end = δω, δω_init = δω, 
    DKS_init =  DKS_init, 
)

solver = pyLLE.LLEsolver(sim=sim, res=res,debug=False)

fig = solver.Analyze(plot=False)
# fig.update_yaxes(range = [-50, 50], title = 'D<sub>int</sub> (GHz)')

solver.disp
solver.sim
solver.res

solver.Setup(verbose = False)
solver.SolveTemporal()
solver.RetrieveData()

import pdb; pdb.set_trace()
fig = solver.PlotCombPower(do_matplotlib=True)
fig.show()
