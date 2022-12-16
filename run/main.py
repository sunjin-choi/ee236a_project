

from lle_experiment import LLEExperiment, ExperimentConfig, draw_nice_canvas, draw_grid_spec, default_font

from matplotlib import pyplot as plt

#def test_save():
#    experiment = LLEExperiment.test_initialize()
#    experiment.run()
#    experiment.save_checkpoint()
#    return experiment
#
#def test_sim():
#    config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=160e-3)
#    experiment = LLEExperiment.initialize_from_config(config)
#    experiment.run()
#    experiment.save_checkpoint()
#    return experiment
#
#def test_load_sim():
#    config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=160e-3)
#    experiment = LLEExperiment.load_checkpoint(config)
#    return experiment


def sweep_Qc():
    Qc_list = [0.8e6, 0.9e6, 1.0e6, 1.1e6, 1.2e6]
    experiment_list = []
    for Qc in Qc_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=Qc, Pin=160e-3)
        experiment = LLEExperiment.initialize_from_config(config)
        experiment.run()
        experiment.save_checkpoint()
        experiment_list.append(experiment)
    return experiment_list

def sweep_Pin():
    Pin_list = [100e-3, 120e-3, 140e-3, 160e-3, 180e-3]
    experiment_list = []
    for Pin in Pin_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=Pin)
        experiment = LLEExperiment.initialize_from_config(config)
        experiment.run()
        experiment.save_checkpoint()
        experiment_list.append(experiment)
    return experiment_list


def load_sweep_Qc():
    Qc_list = [0.8e6, 0.9e6, 1.0e6, 1.1e6, 1.2e6]
    experiment_list = []
    for Qc in Qc_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=Qc, Pin=160e-3)
        experiment = LLEExperiment.load_checkpoint(config)
        experiment_list.append(experiment)
    return experiment_list

def load_sweep_Pin():
    Pin_list = [100e-3, 120e-3, 140e-3, 160e-3, 180e-3]
    experiment_list = []
    for Pin in Pin_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=Pin)
        experiment = LLEExperiment.load_checkpoint(config)
        experiment_list.append(experiment)
    return experiment_list




def sweep_Qc_overcouple():
    Qc_list = [1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 10e5]
    experiment_list = []
    for Qc in Qc_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=Qc, Pin=500e-3)
        experiment = LLEExperiment.initialize_from_config(config)
        experiment.run()
        experiment.save_checkpoint()
        experiment_list.append(experiment)
    return experiment_list

def load_sweep_Qc_overcouple():
    Qc_list = [1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 10e5]
    experiment_list = []
    for Qc in Qc_list:
        config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=Qc, Pin=500e-3)
        experiment = LLEExperiment.load_checkpoint(config)
        experiment_list.append(experiment)
    return experiment_list


def run():
    sweep_Qc()
    sweep_Pin()

def run_overcouple():
    sweep_Qc_overcouple()


def plot():
    experiment_Pin = load_sweep_Pin()
    experiment_Qc = load_sweep_Qc()

    fig, ax = draw_nice_canvas()
    fig = experiment_Pin[0].solver.Analyze(plot=True)
    fig.axes[0].set_xlim([100, 400])
    fig.axes[0].set_ylim([-200, 100])

    spectra_idx = 3600

    fig, ax = draw_nice_canvas()
    label = f"{experiment_Pin[3].main_params.Pin*1e3:.0f} mW"
    fig, ax = experiment_Pin[3].add_axes_comb_power(fig, ax, label, 3/5)

    ax.annotate(xy=(3000, 4.9), text="Single Soliton Step", arrowprops=dict(arrowstyle="->"), xytext=(3000, 10.9))
    ax.annotate(xy=(1931, 16), text="Modulation Instability", arrowprops=dict(arrowstyle="->"), xytext=(1550, 10.9))
    ax.annotate(xy=(971, 6), text="Primary Comb", arrowprops=dict(arrowstyle="->"), xytext=(400, 10.9))
    
    fig.savefig("./figure/dynamics.png", dpi=300, bbox_inches="tight")
    # fig.show()
    

    eff_Pin = []
    for i in range(len(experiment_Pin)):
        eff_Pin.append(experiment_Pin[i].get_conversion_efficiency_from_spectra(spectra_idx))

    eff_Qc = []
    for i in range(len(experiment_Qc)):
        eff_Qc.append(experiment_Qc[i].get_conversion_efficiency_from_spectra(spectra_idx))

    fig, ax = draw_nice_canvas(2, 1)
    ax[0].plot([experiment_Pin[i].main_params.Pin*1e3 for i in range(len(experiment_Pin))], eff_Pin, "o-")
    ax[0].set_xlabel("Input power (mW)", **default_font)
    ax[0].set_ylabel("Conversion Efficiency [%]", **default_font)
    ax[0].set_title("Conversion Efficiency vs. Input Power", **default_font)

    ax[1].plot([experiment_Qc[i].main_params.Qc*1e-6 for i in range(len(experiment_Qc))], eff_Qc, "o-")
    ax[1].set_xlabel("Qc [x1e6]", **default_font)
    ax[1].set_ylabel("Conversion Efficiency [%]", **default_font)
    ax[1].set_title("Conversion Efficiency vs. Qc (Qi=1e6)", **default_font)
    fig.savefig("./figure/conversion_efficiency.png", dpi=300, bbox_inches="tight")
    # fig.show()


    fig, ax = draw_grid_spec()
    for i in range(len(experiment_Pin)):
        label = f"{experiment_Pin[i].main_params.Pin*1e3:.0f} mW"
        fig, ax[0] = experiment_Pin[i].add_axes_comb_power(fig, ax[0], label, i/len(experiment_Pin))

    for i in range(len(experiment_Qc)):
        label = f"{experiment_Qc[i].main_params.Qc*1e-6:.1f}e6 (Qi=1e6)"
        fig, ax[1] = experiment_Qc[i].add_axes_comb_power(fig, ax[1], label, i/len(experiment_Qc))

    for i in range(len(experiment_Pin)):
        label = f"{experiment_Pin[i].main_params.Pin*1e3:.0f} mW"
        fig, ax[2] = experiment_Pin[i].add_axes_comb_spectra(spectra_idx, fig, ax[2], label, i/len(experiment_Pin))

    for i in range(len(experiment_Qc)):
        label = f"{experiment_Qc[i].main_params.Qc*1e-6:.1f}e6 (Qi=1e6)"
        fig, ax[3] = experiment_Qc[i].add_axes_comb_spectra(spectra_idx, fig, ax[3], label, i/len(experiment_Qc))

    ax[2].yaxis.set_label_position("right")
    ax[2].yaxis.tick_right()
    ax[3].yaxis.set_label_position("right")
    ax[3].yaxis.tick_right()

    fig.savefig("./figure/dynamics_and_spectra.png", dpi=300, bbox_inches="tight")
    # fig.show()


    fig, ax = draw_nice_canvas(1, 1)
    fig, ax = experiment_Pin[0].add_axes_dispersion(fig, ax, "Dispersion")
    ax.set_xlim([200, 400])
    ax.set_ylim([-50, 50])
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Dispersion Dint (GHz)")
    ax.set_title("Dispersion vs. Frequency")
    fig.savefig("./figure/example_dispersion.png", dpi=300, bbox_inches="tight")
    # fig.show()


def plot_overcouple():

    experiment_Qc = load_sweep_Qc_overcouple()
    spectra_idx = 3600

    fig, ax = draw_nice_canvas(2, 1)
    for i in range(len(experiment_Qc)):
        label = f"{experiment_Qc[i].main_params.Qc*1e-5:.1f}e5 (Qi=1e6)"
        fig, ax[0] = experiment_Qc[i].add_axes_comb_power(fig, ax[0], label, i/len(experiment_Qc))

    ax[0].set_title("Comb formation vs. Qc (Qi=1e6, Pin=500mW)", **default_font)

    eff_Qc = []
    for i in range(len(experiment_Qc)):
        eff_Qc.append(experiment_Qc[i].get_conversion_efficiency_from_spectra(spectra_idx))
    
    ax[1].plot([experiment_Qc[i].main_params.Qc*1e-6 for i in range(len(experiment_Qc))], eff_Qc, "o-")
    ax[1].set_xlabel("Qc [x1e6]", **default_font)
    ax[1].set_ylabel("Conversion Efficiency [%]", **default_font)
    ax[1].set_title("Conversion Efficiency vs. Qc (Qi=1e6, Pin=500mW)", **default_font)
    fig.show()

    fig.savefig("./figure/overcouple.png", bbox_inches="tight")


if __name__ == "__main__":
    # run()
    # run_overcouple()
    plot()
    plot_overcouple()

