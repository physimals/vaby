"""
Example using a hybrid model structure (1 cortical hemisphere + subcortical WM)
and simulated multi-PLD ASL with simulated variable CBF activation on cortex
"""

import os
import sys 
import argparse

import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt 
import seaborn as sns

import vaby

cli = argparse.ArgumentParser()
cli.add_argument("--outdir", help="Output directory")
cli.add_argument("--mask", help="Nifti file containing analysis mask", default="sphere_mask.nii.gz")
cli.add_argument("--cort-inner", help="Gifti file containing cortex inner surface", default="sphere_insurf.surf.gii")
cli.add_argument("--cort-outer", help="Gifti file containing cortex outer surface", default="sphere_outsurf.surf.gii")
cli.add_argument("--cort-inflated", help="Gifti file containing cortex inflated surface", default="sphere_insurf.surf.gii")
cli.add_argument("--projector", help="Pre-computed projection file", default="sphere_proj.h5")
cli.add_argument("--wm-pvs", help="Nifti file containing WM partial volumes", default="sphere_wm.nii.gz")
cli.add_argument("--show", help="what kind of plot to show", default="surface")
opts = cli.parse_args()

model_structures = [
    {
        "name" : "L",
        "type" : "CorticalSurface",
        "white" : opts.cort_inner,
        "pial" : opts.cort_outer,
        "projector" : opts.projector,
        #"save-projector" : "hybrid_example_projector.h5",
    },
    {
        "name" : "WM",
        "type" : "PartialVolumes",
        "vol_data" : opts.wm_pvs,
        "mask" : opts.mask,
    },
]

# Create data model for simulating data. Note that the acquisition data is just
# used to define the acquisition data space so doesn't need to be a proper timeseries
data_model = vaby.DataModel(opts.wm_pvs, mask=opts.mask, model_structures=model_structures)

from toblerone.classes import Surface
inflated = Surface(opts.cort_inflated, "inflated")
inflated = inflated.transform(data_model.model_space.parts[0].projector.spc.world2vox)

cbf_ctx_true = nib.load(os.path.join(opts.outdir, 'true_ftiss_L.func.gii')).darrays[0].data
cbf_ctx_output = nib.load(os.path.join(opts.outdir, 'mean_ftiss_L.func.gii')).darrays[0].data
act_ctx_true = nib.load(os.path.join(opts.outdir, 'hybrid_example_asl_data_clean_L.func.gii')).darrays[0].data
act_ctx_true = np.mean(act_ctx_true, axis=1)
act_noisy = nib.load(os.path.join(opts.outdir, 'input_data.nii.gz')).get_fdata().astype(np.float)
mask = nib.load(opts.mask).get_fdata() > 0
act_noisy = act_noisy[mask]
act_noisy = data_model.data_to_model(act_noisy, pv_scale=True)
act_ctx_noisy = data_model.model_space.split(act_noisy)['L']
act_ctx_noisy = np.mean(act_ctx_noisy, axis=1)
max_val = np.max(act_ctx_noisy)

if opts.show == "surface":
    # Plot true and predicted CBF
    faces = 3 * np.ones((inflated.tris.shape[0], 4), dtype=int)
    faces[:,1:] = inflated.tris 

    plotter = pv.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plotter.add_text("Estimated CBF", font_size=10)
    plotter.add_mesh(pv.PolyData(inflated.points, faces=faces), scalars=cbf_ctx_output, clim=(0, 100), scalar_bar_args={'title': 'Estimated CBF'}, show_scalar_bar=True)
    plotter.add_axes(interactive=True)

    plotter.subplot(0, 1)
    plotter.add_text("True CBF", font_size=10)
    plotter.add_mesh(pv.PolyData(inflated.points, faces=faces), scalars=cbf_ctx_true, clim=(0, 100), scalar_bar_args={'title': 'True CBF'}, show_scalar_bar=True)
    plotter.add_axes(interactive=True)

    plotter.subplot(1, 0)
    plotter.add_text("True activation", font_size=10)
    plotter.add_mesh(pv.PolyData(inflated.points, faces=faces), scalars=act_ctx_true, clim=(0, max_val), scalar_bar_args={'title': 'True activation'}, show_scalar_bar=True)
    plotter.add_axes(interactive=True)

    plotter.subplot(1, 1)
    plotter.add_text("Noisy activation", font_size=10)
    plotter.add_mesh(pv.PolyData(inflated.points, faces=faces), scalars=act_ctx_noisy, clim=(0, max_val), scalar_bar_args={'title': 'Noisy activation'}, show_scalar_bar=True)
    plotter.add_axes(interactive=True)

    # Display the window
    plotter.show()
elif opts.show == "histogram":
    # Plot histogram comparison of CBF on cortex
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(3,3), sharex=True, sharey=True)

    sns.kdeplot(x=cbf_ctx_true, label='truth', ax=ax)
    sns.kdeplot(x=cbf_ctx_output, ax=ax, label='inferred')

    ax.legend()
    ax.set_xlim(0, 120)

    fig.suptitle('Hybrid mode')
    plt.show()
else:
    raise RuntimeError("Unrecognized visualisation: %s" % opts.show)
