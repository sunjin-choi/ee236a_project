

from lle_experiment import LLEExperiment, ExperimentConfig

def test_save():
    experiment = LLEExperiment.default_initializer()
    experiment.run()
    experiment.save_checkpoint("test")
    return experiment

def test_load():
    experiment = LLEExperiment.load_checkpoint("test")
    return experiment


def test_sim():
    config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=160e-3)
    experiment = LLEExperiment.initialize_from_config(config)
    experiment.run()
    experiment.save_checkpoint()
    return experiment

def test_load_sim():
    config = ExperimentConfig(R=23e-6, Qi=1e6, Qc=1e6, Pin=160e-3)
    experiment = LLEExperiment.load_checkpoint(config)
    return experiment


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


if __name__ == "__main__":
    experiment = test_sim()
    # experiment = test_load_sim()

    # experiment_Qc = sweep_Qc()
    # experiment_Pin = sweep_Pin()

    # experiment_Pin = load_sweep_Pin()
    # experiment_Qc = load_sweep_Qc()

    fig = experiment.solver.Analyze(plot=True, plottype="all")
    # fig.axes[0].set_xlim([100, 400])
    # fig.axes[0].set_ylim([-200, 100])
    import pdb; pdb.set_trace()
