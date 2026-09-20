# Burgers' Equation: Finite Volume Solver

This repository contains a numerical implementation for solving the one-dimensional Burgers' equation using a first-order finite volume method. A detailed explanation of the theory is described in report that can be found here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22859287.svg)](https://doi.org/10.5281/zenodo.22859287)

The spatial domain is discretized on a uniform grid, and the governing equation is integrated over each finite volume cell. The solution is represented by cell averages, and an upwind flux is used to approximate the flux across cell interfaces. For a uniform one-dimensional grid, each cell has a volume equal to the spatial grid spacing, which leads to a straightforward finite volume discretization.

Time integration is performed using the explicit Euler method, resulting in an iterative scheme where the solution in each cell is updated based on the flux difference between neighboring cells.

As a further development of the project, the finite volume discretization will be reformulated as a linear system of equations. This system will be solved using GMRES, with a multigrid method used as a preconditioner to improve the efficiency and convergence of the iterative solver.

The overall goal is to investigate the numerical solution of Burgers' equation while exploring the use of iterative linear solvers and multigrid preconditioning in the resulting discretized system.

## Files

* **GMRES.py** — A homemade implementation of the GMRES solver. The solver is currently a basic implementation and will be improved for better efficiency in the future.
* **KrylovSpace.py** — Constructs a Krylov space with an orthonormal basis for a given linear system of equations.
* **BurgersEquation.py** — Defines the one-dimensional Burgers' equation in Newton form.
* **NewtonsMethods.py** — Implements an iterative Newton's method solver for solving zero-function problems using the Jacobian.
* **MultiGridMethod.py** — Implements the multigrid method, which is used as a preconditioner for the iterative solver.
* **OneStep.ipynb** — A Jupyter notebook demonstrating how to solve one time step of the Burgers' equation using the methods implemented in this repository.

## Environment Setup

This project uses **Conda** for environment and package management. The required packages and versions are specified in `environment.yml`.

### 1. Create the environment

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-name>
```

Create the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate numerics
```



