![LOGO](https://github.com/DeepWave-KAUST/DiffusionDAS_Denoising/blob/main/asset/DiffDAS.png)

Reproducible material for: **Self-supervised Diffusion Model for Distributed Acoustic Sensing Data Denoising - Omar M. Saad and Tariq Alkhalifah**


# Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data (uploaded to the restricted area);
* :open_file_folder: **Prepare_CWT_MATLAB**: folder containing a demo example for preparing the CWT using MATLAB. It is a more stable version of the 2D CWT using MATLAB.
* :open_file_folder: **py_cwt2d**: includes CWT package for the 2D CWT using Python.


## Notebooks
The following notebooks are provided (**using MATLAB to prepare the CWT scales**):

- :orange_book: ``Diffusion_DAS_Denoising_FORGE.ipynb``: notebook performing self-supervised Diffusion DAS denoising for FORGE example;
- :orange_book: ``Diffusion_DAS_Denoising_SAFOD.ipynb``: notebook performing self-supervised Diffusion DAS denoising for SAFOD example;
- :orange_book: ``Diffusion_DAS_Denoising_Greece.ipynb``: notebook performing self-supervised Diffusion DAS denoising for Greece example;

If you prefer not to use MATLAB for preparing the CWT, the following notebook (Python version) is provided to generate the CWT scales. **However, please note that the Python CWT implementation is not optimal, as it tends to introduce more signal leakage compared to the more stable and reliable CWT computation available in MATLAB**
- :orange_book: ``Diffusion_DAS_Denoising_FORGE_PythonCWT.ipynb``: notebook for self-supervised diffusion-based DAS denoising on the FORGE example, using 2D CWT scales computed with the Python package.


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal, you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate DiffDAS
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
Omar M. Saad and Tariq Alkhalifah (2026) Self-supervised Diffusion Model for Distributed Acoustic Sensing Data Denoising, Geophysical Prospective.

