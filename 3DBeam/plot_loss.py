import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from DemBeam3D import plot_losses
import seaborn as sns
sns.set()

current_path = Path().cwd().resolve()
arrays_path = current_path / 'stored_arrays'

losses_lrs_nl = np.load(arrays_path / 'losses_lrs_nl.npy')
# print(losses_lrs_nl)
losses_lrs_nl += np.abs(np.min(losses_lrs_nl[~np.isnan(losses_lrs_nl)]))
# print(losses_lrs_nl)
# print(np.min(losses_lrs_nl))

plot_losses(losses, )