
import os 
import sys
import inspect
from typing import NamedTuple

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

from matplotlib import pyplot as plt


default_font = {'fontname': 'serif', 'fontsize': 16}

def draw_nice_canvas(num_x=1, num_y=1):
    """
    Draw a beautiful gridded canvas and return fig, ax
    """
    fig, ax = plt.subplots(num_y, num_x, figsize=(20, 10))

    if isinstance(ax, np.ndarray):
        for i in range(num_x):
            for j in range(num_y):
                ax[i*num_y + j].grid()
    else:
        ax.grid()
    # # set font to academic paper best fit
    # plt.rc('font', family='serif', size=16)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16

    return fig, ax

def draw_grid_spec():
    """
    Draw a beautiful gridded canvas and return fig, ax
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    ax = []

    ax.append(fig.add_subplot(gs[0, 0:2]))
    ax.append(fig.add_subplot(gs[1, 0:2]))
    ax.append(fig.add_subplot(gs[0, 2]))
    ax.append(fig.add_subplot(gs[1, 2]))

    for j in range(4):
        ax[j].grid()

    # plt.rc('font', family='serif', size=16)

    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 16

    return fig, ax


class ExperimentConfig(NamedTuple):
    R: float
    Qi: float
    Qc: float
    Pin: float

    def __str__(self):
        return f"ExperimentConfig(R={self.R}, Qi={self.Qi}, Qc={self.Qc}, Pin={self.Pin})"

    def __repr__(self):
        return f"ExperimentConfig(R={self.R}, Qi={self.Qi}, Qc={self.Qc}, Pin={self.Pin})"


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
                dispfile="./config/RW1000_H430.csv"
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
                dispfile="./config/TestDispersion.csv"
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
                dispfile="./config/RW1000_H430.csv"
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

    def add_axes_comb_power(self, fig, ax, label, cnum):
        fig, ax = self.PlotCombPower(fig, ax, label, cnum, which='comb', xaxis='steps')
        return fig, ax

    def add_axes_comb_spectra(self, ind, fig, ax, label, cnum):
        fig, ax = self.PlotCombSpectra(ind, fig, ax, label, cnum)
        return fig, ax

    def add_axes_dispersion(self, fig, ax, label):
        fig, ax = self.PlotDispersion(fig, ax, label)
        return fig, ax

    def get_conversion_efficiency_from_spectra(self, ind):
        comb_spectra = np.abs(self.solver.sol.Ecav[:, ind]**2)
        comb_spectra_sort = np.flip(np.sort(comb_spectra))

        return np.sum(comb_spectra_sort[1:]) / self.main_params.Pin * 1e2



    def infrared(self, cnum):
        """Returns a color for a given infrared value.
        The color palette is a list of colors that can be used to represent different data.

        :param cnum: The index of the color to be returned, usually normalized to 1
        :return: The color at the index of the color palette
        """
        return plt.cm.get_cmap('plasma')(cnum)

    def PlotCombPower(self, fig, ax, label, cnum, which = 'all', xaxis = 'steps'):
        tr = []
        if xaxis.lower() == 'steps':
            x = np.linspace(0,4999,5000)
            # xlabel = 'LLE steps'
            xlabel = 'Laser Scan'
        elif xaxis.lower() == 'detuning':
            x = 1e-9*self.solver._sol['detuning']/(2*np.pi)
            xlabel = 'Detuning (GHz)'

        if which.lower() == 'all':
            ax.plot(x, self.solver.sol.Pcav)
            ax.plot(x, self.solver.sol.Pwg)
            ax.plot(x, self.solver.sol.Pcomb)

        if which.lower() == 'comb':
            ax.plot(x, self.solver.sol.Pcomb*1e3, label=label, color=self.infrared(cnum))
        if which.lower() == 'waveguide':
            ax.plot(x, self.solver.sol.Pwg*1e3, label=label, color=self.infrared(cnum))
        if which.lower() == 'cavity':
            ax.plot(x, self.solver.sol.Pcav*1e3, label=label, color=self.infrared(cnum))

        ax.legend()
        ax.set_xlabel(xlabel, **default_font)
        ax.set_ylabel('Power (mW)', **default_font)

        return fig, ax

    def PlotCombSpectra(self, ind, fig, ax, label, cnum, style = '', xaxis = 'freq', where = 'waveguide', floor = -100):
        
        if xaxis.lower() == 'frequencies' or xaxis.lower() == 'freq':
            x = self.solver.sol.freq*1e-12
            xlabel = 'Frequency (THz)'
        elif xaxis.lower() == 'modes':
            x = np.arange(self.solver._sim['mu_sim'][0],self.solver._sim['mu_sim'][-1])  - self.solver._sim['ind_pmp'][0]
            xlabel = 'Mode Numbers'

        if where == 'waveguide' or where =='wg':
            y = 10*np.log10(np.abs(self.solver.sol.Ewg[:,ind])**2)+30
            name = ['Waveguide']
        if where == 'cavity' or where =='cav':
            y = 10*np.log10(np.abs(self.solver.sol.Ecav[:,ind])**2)+30
            name = ['Cavity']
        if where == 'both':
            y = 10*np.log10(np.abs(self.solver.sol.Ecav[:,ind])**2)+30
            y2 = 10*np.log10(np.abs(self.solver.sol.Ewg[:,ind])**2)+30
            name = ['Cavity', 'Waveguide']

        if style == 'comb':
            x_ = np.zeros(x.size*3)
            x_[::3] = x
            x_[1::3] = x
            x_[2::3] = x
            x = x_

            y_ = np.zeros(y.size*3)
            y_[::3] = floor
            y_[1::3] = y
            y_[2::3] = floor
            y = y_

            if where == 'both':
                y2_ = np.zeros(y2.size*3)
                y2_[::3] = floor
                y2_[1::3] = y2
                y2_[2::3] = floor
                y2 = y2_


        ax.plot(x, y, label = label, color = self.infrared(cnum))
        if where == 'both':
            ax.plot(x, y2, label = name[1])

        # # draw vertical line at pump frequency
        # ax.axvline(x=self.sim.f_pmp[0]*1e-12, color='k', linestyle='--', linewidth=1)
        
        ax.legend()
        ax.set_xlabel(xlabel, **default_font)
        ax.set_ylabel('Power (dBm)', **default_font)

        self.solver._plotSpecta = True
        self.solver._indSpectra = ind

        return fig, ax



    def PlotDispersion(self, fig, ax, label):
        for ii in range(len(self.solver._analyze.fpmp)-1):
            μfit = self.solver._analyze.μfit[ii] #+ self.solver._analyze.ind_pmp_fit[ii]
            μsim = self.solver._analyze.μsim[ii] #+ self.solver._analyze.ind_pmp_sim[ii]
            dν_fit = np.arange(μfit[0], μfit[-1]+1)*self.solver._analyze.D1[0]/(2*np.pi)
            dν_sim = np.arange(μsim[0], μsim[-1]+1)*self.solver._analyze.D1[0]/(2*np.pi)
            ν0 = self.solver._analyze.fpmp[ii]
            rf = self.solver._analyze.rf
            rf_fit = (ν0+dν_fit)
            rf_sim = (ν0+dν_sim)

            ax.plot(rf*1e-12,
                   self.solver._analyze.Dint[ii]*1e-9/(2*np.pi),
                   'o',ms= 3,
                   label = 'FEM Simulation')


            ax.plot(rf_fit*1e-12,
                   self.solver._analyze.Dint_fit[ii]*1e-9/(2*np.pi),
                   '--',
                   label = 'Fit')
            ax.plot(rf_sim*1e-12,
                   self.solver._analyze.Dint_sim[ii]*1e-9/(2*np.pi),
                   label = 'LLE simulation')

            if self.solver._analyze.plottype.lower() == 'sim':
                ax.plot(rf_sim*1e-12,
                       self.solver._analyze.Dint_sim[ii]*1e-9/(2*np.pi),
                       label = 'LLE simulation')

            if self.solver._analyze.plottype.lower() == 'fit':
                ax.plot(rf_fit*1e-12,
                       self.solver._analyze.Dint_fit[ii]*1e-9/(2*np.pi),
                       label = 'Fit')

        # draw vertical line at pump frequency
        ax.axvline(x=self.sim.f_pmp[0]*1e-12, color='k', linestyle='--', linewidth=1)

        ax.legend()
        ax.set_xlim([200, 400])
        ax.set_ylim([-50, 50])
        return fig, ax
