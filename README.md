# Burgers' Equation: Finite Volume Solver

This repository contains a numerical implementation for solving the one-dimensional Burgers' equation using a first-order finite volume method.

The spatial domain is discretized on a uniform grid, and the governing equation is integrated over each finite volume cell. The solution is represented by cell averages, and an upwind flux is used to approximate the flux across cell interfaces. For a uniform one-dimensional grid, each cell has a volume equal to the spatial grid spacing, which leads to a straightforward finite volume discretization.

Time integration is performed using the explicit Euler method, resulting in an iterative scheme where the solution in each cell is updated based on the flux difference between neighboring cells.

As a further development of the project, the finite volume discretization will be reformulated as a linear system of equations. This system will be solved using GMRES, with a multigrid method used as a preconditioner to improve the efficiency and convergence of the iterative solver.

The overall goal is to investigate the numerical solution of Burgers' equation while exploring the use of iterative linear solvers and multigrid preconditioning in the resulting discretized system.
