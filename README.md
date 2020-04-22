# Solvent hydrodynamics and membrane kinetics for coarse-grained models

This repository contains various codes for the calculation of anisotropic diffusion tensors based on first-principle hydrodynamics. The method is intended to be incorporated in solvent-free coarse-grained membrane models. The theoretical background for the presented calculations are given in [1, 2]. We have demonstrated this approach to reproduce realistic membrane kinetics in [1], using the mesoscopic membrane model described in [3].

> hydrodynamics

Contains Jupyter notebooks to demonstrate how the hydrodynamic response of the fluid domain to test forces and displacements is obtained.  Three notebooks are included for single planar membranes, parallel planar membranes, and spherical vesicles.

> diffusion_tensors

The Python codes to employ the hydrodynamic response theory and calculate the out-of-plane components of friction and diffusion tensors and the magnitude of hydrodynamic interactions across the membrane for the three cases of single planar membranes, parallel planar membranes, and spherical vesicles. Expamples for how to use these codes are given in the Jupyter notebook `compiled_diffusion_friction_tensor`.


## References

[1] Sadeghi, M. & Noé, F. (2020). "Large scale simulation of biomembranes incorporating realistic kinetics into coarse-grained models" _Nat Commun_, preprint at https://www.biorxiv.org/content/10.1101/815571v1

[2] Sadeghi, M. & Noé, F. (2019). "First-principle hydrodynamics and kinetics for solvent-free coarse-grained membrane models" _arXiv_, 1909.02722. https://arxiv.org/abs/1909.02722

[3] Sadeghi, M., Weikl, T. R. & Noé, F. (2018). "Particle-based membrane model for mesoscopic simulation of cellular dynamics" _J Chem Phys_, 148(4), 044901. https://doi.org/10.1063/1.5009107


