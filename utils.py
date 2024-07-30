import os
import mne
from tqdm import tqdm
from pmf import *
import matplotlib.pyplot as plt
import numpy as np

def calculate_rvs(raw,W,M,N,r,tmin = -0.01,tmax = 2.5):

    #FOR DIPOLE CALCULATIONS ONLY
    subjects_dir="~/mne_data/MNE-fsaverage-data"
    subject="fsaverage"

    fs_dir = mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir,verbose=False)

    fname_bem = os.path.join(fs_dir, "bem" , "fsaverage-5120-5120-5120-bem-sol.fif")
    fname_trans = os.path.join(fs_dir, "bem" , "fsaverage-trans.fif")
    fname_surf_lh = os.path.join(fs_dir, "surf" , "lh.white")

    raw_avg = raw.copy()
    raw_avg.load_data(verbose=False)
    raw_avg.set_eeg_reference(verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw_avg, events, event_id=event_id['square'], tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True,verbose=False )
    # cov = mne.compute_covariance(epochs) 
    cov = mne.make_ad_hoc_cov(epochs.info, verbose=False)

    #Calculate PMF dipoles for all
    
    dip_pmf_rvs = []
    dip_pmf_gofs = []
    # dip_pmf_list = []

    for i in tqdm(range(r), desc="Calculating RVs"):
        # print(i)

        component_pmf = mne.EvokedArray(W.T[i, :].reshape(M,1), info=epochs.info, verbose=False)
        dip_pmf, _ = mne.fit_dipole(component_pmf, cov, fname_bem, fname_trans, verbose=False)
        # dip_pmf_list.append(dip_pmf)

        fwd_pmf, stc_pmf = mne.make_forward_dipole(dip_pmf, fname_bem, raw.info, fname_trans, verbose=False)
        pred_evoked_pmf = mne.simulation.simulate_evoked(fwd_pmf, stc_pmf, raw.info, cov=None, nave=np.inf, verbose=False)
        residual_variance = np.var(component_pmf.data - pred_evoked_pmf.data) / np.var(component_pmf.data)

        dip_pmf_rvs.append(residual_variance)
        dip_pmf_gofs.append(dip_pmf.gof[0])

    return dip_pmf_rvs, dip_pmf_gofs

##GRID PLOT FUNC
def pmf_plot(r, W, raw, dip_pmf_rvs=None, treshold=None):
    # Assuming r is defined and raw.info is available
    # Determine the number of rows and columns for the grid

    if treshold is not None:
        indices_below_treshold_pmf = np.where(np.array(dip_pmf_rvs) < treshold)[0]
        r = len(indices_below_treshold_pmf)
    else:
        indices_below_treshold_pmf =range(r)

    a, b, i = 1, r, 0
    while a < b:
        i += 1
        if r % i == 0:
            a = i
            b = r//a
    
    n_cols = int(a)
    n_rows = int(b)

    
    if a == r or b == r:
        c = np.ceil(np.sqrt(r))
        d=1
        
        while c*d < r:
            d = d + 1

        n_cols = int(c)
        n_rows = int(d)


    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    for i in range(r):
        id = indices_below_treshold_pmf[i]
        # Plot topomap on the respective subplot
        mne.viz.plot_topomap(W.T[id, :], raw.info, axes=axes[i], show=False)

        if dip_pmf_rvs is not None:
            axes[i].set_title(f'Source {id}: \n RV: {dip_pmf_rvs[id]:.2f}')
        else:
            axes[i].set_title(f'Source {id}')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
