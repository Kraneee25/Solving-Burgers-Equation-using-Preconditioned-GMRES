# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 08:34:33 2025
@author: krane
"""
from NewtonsMethod import NewtonsMethod
import matplotlib.pyplot as plt
from MultiGridMethod import MultiGrid
import numpy as np

class BurgerNewtonForm ():
    
    def __init__(self, dt,  numcells, u0_func, domain=[0,2]):
        self.dt          = dt
        self.u0_func     = u0_func
        self.domain      = domain  
        self.centers = self.cell_centers(numcells)
        self._u0     = u0_func(self.centers)
        self.u0_func = u0_func
        
        #Statistics
        self.jacob_evals = 0
        self.new_jacob   = 0
        self.F_evals     = 0
        
    def cell_centers(self, n_cells):
        """Uses linspace to create the cell centers. 
        We place the dofs on the centers for simple MG restriction"""
        self.dx = (self.domain[1] - self.domain[0]) / n_cells
        return np.linspace(self.domain[0] + self.dx/2, 
                           self.domain[1] - self.dx/2, n_cells)
    
    def analytical_flux (self, u ):          
        return 0.5*u**2

    def getNumericalFlux (self, uvals):
        "Made for periodic boundary condition"
        F     = np.zeros(len(uvals)+1)
        F[0]  = self.analytical_flux(uvals[-1])
        F[1:] = self.analytical_flux(uvals)
        return F
            
    def zeroProblem (self, u):
        "Non-linear root problem that can be adapted to any grid resolution."
        self.F_evals += 1
        if len(u) != len(self._u0): # hand-wavy
            self.centers = self.cell_centers(len(u))
            self._u0  = self.u0_func(self.centers )
            
        dummy_flux_array = self.dt/(2/len(u)) * self.getNumericalFlux(u)
        numerical_flux  = dummy_flux_array[1: ]  #i       
        numerical_flux -= dummy_flux_array[:-1]  #i-1
        Fuk1 = u - self._u0 + numerical_flux
        
        return Fuk1
    
    def inexactJacobian_operator (self, uk):
        "FDM approximation of the Jacobian."
        self.new_jacob += 1
        def Fprime(delta_x):
            self.jacob_evals += 1
            if np.allclose(delta_x, 0.0, rtol=0.0, atol=1e-15):
                return np.zeros_like(delta_x)
            epsilon = np.sqrt(np.finfo(float).eps) / np.linalg.norm(delta_x)
            
            return ( self.zeroProblem(uk + epsilon*delta_x) \
                    - self.zeroProblem(uk) )/epsilon
        
        return Fprime
    
    def exactJacobian (self, uk):
        n = len(uk)
        B = np.zeros((n, n))
        # diagonal
        np.fill_diagonal(B, uk)
        # subdiagonal
        B[1:, :-1] -= np.diag(uk[:-1])
        # special corner coupling
        B[0, -1] = -uk[-1]
        
        return np.ones_like(uk) + self.dt/(2/len(uk)) * B
    
    def check_solution (self, u):
        def discretized_model (u):
            return (u - self._u0)/self.dt + (1/(2/len(u)))*(self.analytical_flux(u) - self.analytical_flux(np.concatenate((np.array([u[-1]]),u[:-1]))))
        assert np.allclose(discretized_model(u), np.zeros_like(u)), f"{np.linalg.norm(discretized_model(u))}"
        print("Solution Satisfied")
        