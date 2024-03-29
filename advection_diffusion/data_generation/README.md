# Installation instructions for forward solver

## REQUIREMENTS
* `Anaconda/Miniconda`
* `pyapprox`
* `fenics`
* `mambda` (optional)

## 1. Anaconda (miniconda also works)
Download and install conda for your OS at https://www.anaconda.com/products/individual

## 2. PyApprox
```
cd <path-to-install-directory>
git clone https://github.com/sandialabs/pyapprox.git  
cd pyapprox
conda env create -f environment.yml
conda activate pyapprox-base
pip install -e .
```

## 3. FEniCS (this could take a while)
```
conda install -c conda-forge fenics
```
**Note**: If the `FEniCS` installation fails at this step, use the `mamba` package manager instead:
```
conda install mamba -n base -c conda-forge
mamba install fenics -c conda-forge
```
## To run the forward solver
* Adjust the PATH in generate_data.py, then run it with its five command line arguments
* OPTIONAL: adjust the PATH in process_data.py, then run it with its single command line argument
