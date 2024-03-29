# Fourier Neural Mappings
Fourier Neural Mappings (FNMs) generalize Fourier Neural Operators (FNOs) by allowing the input space and/or the output space to be finite-dimensional (instead of both being purely infinite-dimensional function spaces as with FNO). This is especially relevant for surrogate modeling tasks in uncertainty quantification, inverse problems, and design optimization, where a finite number of parameters or quantities of interest (QoIs) characterize the inputs and/or outputs.

In particular, FNMs are able to accommodate
* **nonlinear functions (V2V)**: *Fourier Neural Networks* mapping vectors to vectors (going through a latent function space in between);
* **nonlinear functionals (F2V)**: *Fourier Neural Functionals* mapping functions to vectors (a.k.a. nonlinear encoders);
* **nonlinear decoders (V2F)**: *Fourier Neural Decoders mapping* vectors to functions; and of course
* **nonlinear operators (F2F)**: *Fourier Neural Operators mapping* functions to functions.

In `fourier-neural-mappings`, the network layers in all four types of mappings above are efficiently implemented (via FFT) in Fourier space in a function-space consistent way. The code defaults to running on GPU, if one is available.

## Installation
The command
```
conda env create -f Project.yml
```
creates an environment called ``fno``. [PyTorch](https://pytorch.org/) will be installed in this step.

Activate the environment with
```
conda activate fno
```
and deactivate with
```
conda deactivate
```

The advection-diffusion example requires additional packages to generate and process the data; please refer to the README instructions within that directory for more details.

## Data
The data may be downloaded at [![DOI](https://data.caltech.edu/badge/DOI/10.22002/r5ga1-55d06.svg)](https://doi.org/10.22002/r5ga1-55d06), which contains three `*.zip` files:
1. advection_diffusion: train and test sets for KLE dimension 2, 20, 1000. 
2. airfoil: deformation map (X,Y) coordinates, pressure field, and control nodes.
3. homogenization: V2V, F2V, V2F, and F2F formats.

The data are stored as PyTorch `*.pt` files, `*.npy` arrays, or pickle `*.pkl` files.

```
Huang, D. Z., Nelsen, N. H., & Trautner, M. (2024). An operator learning perspective on parameter-to-observable maps [Data set]. CaltechDATA. https://doi.org/10.22002/r5ga1-55d06. Feb. 12, 2024.
```

## References
The main reference that explains the Fourier Neural Mappings framework is the paper ``[An operator learning perspective on parameter-to-observable maps](https://arxiv.org/abs/2402.06031)'' by Daniel Zhengyu Huang, Nicholas H. Nelsen, and Margaret Trautner. Other relevant references include:
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://arxiv.org/abs/2207.05209)
- [Learning Homogenization for Elliptic Operators](https://arxiv.org/abs/2306.12006)

## Citing
If you use `fourier-neural-mappings` in an academic paper, please cite the main reference ``An operator learning perspective on parameter-to-observable maps'' as follows:
```
@article{huang2024fnm,
  title={An operator learning perspective on parameter-to-observable maps},
  author={Huang, Daniel Zhengyu and Nelsen, Nicholas H and Trautner, Margaret},
  journal={arXiv preprint arXiv:2402.06031},
  year={2024}
}
```

## Contribute
You are welcome to submit an issue for any questions related to `fourier-neural-mappings` or to contribute to the code by submitting pull requests.

## Acknowledgements
The FNO implementation in `fourier-neural-mappings` is adapted from the [original implementation](https://github.com/neuraloperator/neuraloperator/tree/master) by Nikola Kovachki and Zongyi Li. The data generation code for the advection-diffusion example was provided by Zachary Morrow. The `matplotlib` formatting used to produce figures is adapted from the [PyApprox package](https://github.com/sandialabs/pyapprox) by John Jakeman.
