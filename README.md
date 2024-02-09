# Fourier Neural Mappings
Fourier Neural Mappings (FNMs) generalize Fourier Neural Operators (FNOs) by allowing the input space and/or the output space to be finite-dimensional (instead of both being purely infinite-dimensional function spaces as with FNO). This is especially relevant for surrogate modeling tasks in uncertainty quantification, inverse problems, and design optimization, where a finite number of parameters or quantities of interest (QoIs) characterize the inputs and/or outputs.

In particular, FNMs are able to accommodate
* `nonlinear functions (V2V)`: *Fourier Neural Networks* mapping vectors to vectors (going through a latent function space in between);
* `nonlinear functionals (F2V)`: *Fourier Neural Functionals* mapping functions to vectors (a.k.a. nonlinear encoders);
* `nonlinear decoders (V2F)`: *Fourier Neural Decoders mapping* vectors to functions; and of course
* `nonlinear operators (F2F)`: *Fourier Neural Operators mapping* functions to functions.

In `fourier-neural-mappings`, the network layers in all four types of mappings above are efficiently implemented (via FFT) in Fourier space in a function-space consistent way.

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

The individual examples may require additional packages to generate and process the data, train the models, etc. Please refer to the README instructions within each directory, if available.

## Data
The data sets are https://data.caltech.edu/records/20091, which contain 8 `*.npy` files:
1. Navier stokes equation : NavierStokes_inputs.npy & NavierStokes_outputs.npy. 
2. Helmholtz equation : Helmholtz_inputs.npy & Helmholtz_outputs.npy. 
3. Structural mechanics equation : StructuralMechanics_inputs.npy & StructuralMechanics_outputs.npy. 
4. Advection equation : Advection_inputs.npy & Advection_outputs.npy. 

The data are stored as nx by ny by ndata arrays (2d problems) or nx by ndata arrays (1d problems).

## Running the example
In the script ``train.py``, assign in the variable ``data_path`` the global path to the data file ``burgers_data_R10.mat``.

The example may then be run as
```
python -u train.py M N J 0 lambda my_path
```
where
* ``M`` is the number of random features,
* ``N`` is the number of training data pairs,
* ``J`` is the desired spatial resolution for training and testing.
* ``lambda`` is the regularization parameter
* ``my_path`` is the output directory

The code defaults to running on GPU, if one is available.

## Contribute
You are welcome to submit an issue for any questions related to `fourier-neural-mappings` or to contribute to the code by submitting pull requests.

## Acknowledgements
The FNO implementation in `fourier-neural-mappings` is adapted from the [original implementation](https://github.com/neuraloperator/neuraloperator/tree/master) by Zongyi Li and Nikola Kovachki. The data generation code for the advection-diffusion example was provided by Zachary Morrow. The `matplotlib` formatting used to produce figures is adapted from the [PyApprox package](https://github.com/sandialabs/pyapprox) by John Jakeman.

## References
The main reference that explains the Fourier Neural Mappings framework is the paper ``An operator learning perspective on parameter-to-observable maps'' by Daniel Zhengyu Huang, Nicholas H. Nelsen, and Margaret Trautner. Other relevant references are now listed:
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Fourier Neural Operator with Learned Deformations for PDEs on General Geometries](https://arxiv.org/abs/2207.05209)
- [Learning Homogenization for Elliptic Operators](https://arxiv.org/abs/2306.12006)

## Citing
If you use `fourier-neural-mappings` in an academic paper, please cite the main reference ``An operator learning perspective on parameter-to-observable maps'' as follows:
```
@article{huang2024fnm,
  title={An operator learning perspective on parameter-to-observable maps},
  author={Huang, Daniel Zhengyu and Nelsen, Nicholas H and Trautner, Margaret},
  journal={arXiv preprint arXiv:2402.XXXX},
  year={2014}
}
```
