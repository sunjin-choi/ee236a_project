
import os 
import sys
import inspect
from dataclasses import dataclass

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


@dataclass(init=True, repr=True)
class ExperimentConfig:
    R: float
    Qi: float
    Qc: float
    Pin: float


class LLEExperiment:

    def __init__(self, solver):
        self.solver = solver

    def __str__(self):
        return self.main_params.__str__()

    def __repr__(self):
        return self.main_params.__repr__()

    @classmethod
    def initialize_with_sim_res(cls, sim, res):
        solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
        return cls(solver)

    @classmethod
    def test_initialize(cls):
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
        return cls(solver)

    @classmethod
    def test_initialize_Cband(cls):
        res = dict(
                R=23e-6, 
                Qi=1e6, 
                Qc=1e6, 
                γ=1.55,
                dispfile="./TestDispersion.csv"
        )

        sim = dict(
            Pin=[160e-3], 
            f_pmp=[191e12],
            φ_pmp=[0], 
            # δω=[None],
            Tscan=1e6,
            μ_sim=[-74, 170],
            μ_fit=[-71, 180],
            δω_init= 2e9 * 2 * np.pi,
            δω_end= -8e9 * 2 * np.pi,
        )

        solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
        return cls(solver)

    @classmethod
    def initialize_from_config(cls, config: ExperimentConfig):
        res = dict(
                R=config.R,
                Qi=config.Qi,
                Qc=config.Qc,
                γ=3.2,
                dispfile="./RW1000_H430.csv"
        )

        sim = dict(
            Pin=[config.Pin],
            f_pmp=[283e12],
            φ_pmp=[0], 
            δω=[None], 
            Tscan=0.7e6,
            μ_sim=[-220, 220],
            δω_init= 1e9 * 2 * np.pi,
            δω_end= -3.5e9 * 2 * np.pi,
        )

        solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
        return cls(solver)


    @classmethod
    def load_checkpoint(cls, config: ExperimentConfig) -> None:
        solver = pkl.load(open(os.path.join("../checkpoint/", f"{config}.pkl"), "rb"))
        return cls(solver)

    @classmethod
    def load_checkpoint_by_fname(cls, fname, path="../checkpoint/") -> None:
        solver = pkl.load(open(os.path.join(path, f"{fname}.pkl"), "rb"))
        return cls(solver)


    @property
    def sim(self):
        return self.solver.sim

    @property
    def res(self):
        return self.solver.res

    @property
    def main_params(self):
        return ExperimentConfig(
            R=self.res.R,
            Qi=self.res.Qi,
            Qc=self.res.Qc,
            Pin=self.sim.Pin[0],
        )
        
    def run(self):

        self.solver.Analyze(plot=False)
        self.solver.Setup(verbose=True)
        self.solver.SolveTemporal()
        self.solver.RetrieveData()

    def save_checkpoint(self) -> None:
        self.solver.SaveResults(f"{self}", "../checkpoint/")

    def save_checkpoint_by_fname(self, fname, path="../checkpoint/") -> None:
        if not os.path.exists(path):
            os.mkdir(path)
        self.solver.SaveResults(fname, path)


    def get_axes_comb_power(self):
        fig = self.solver.PlotCombPower(do_matplotlib=True, which='comb', xaxis='detuning')
        return fig

    def get_axes_comb_spectra(self, ind):
        fig = self.solver.PlotCombSpectra(ind=ind, do_matplotlib=True)
        return fig
