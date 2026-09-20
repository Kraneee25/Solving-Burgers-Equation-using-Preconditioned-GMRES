# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 08:34:33 2025
@author: krane
"""
from NewtonsMethod import NewtonsMethod
import matplotlib.pyplot as plt
from MultiGridMethod import MultiGrid
import os
import matplotlib as mpl

mpl.rcParams.update({
    "image.cmap": "turbo",
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-3, 3),
    "axes.formatter.use_mathtext": True,
    
    "lines.linewidth": 2.5,    # increase default line width

    "axes.titlesize": 25,      # title size
    "axes.labelsize": 16,      # x/y label size
    "xtick.labelsize": 15,     # x tick size
    "ytick.labelsize": 15,     # y tick size

    "figure.dpi": 120,
    "savefig.dpi": 300,
})

import numpy as np

class BurgerNewtonForm ():
    
    def __init__(self, dt,  numcells, u0_func, domain=[0,2]):
        self.dt          = dt
        self.u0_func     = u0_func
        self.domain      = domain  
        self.centers = self.cell_centers(numcells)
        self._u0     = u0_func(self.centers)
    
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
            self._u0  = u0_func(self.centers )
            
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
            return (u - self._u0)/self.dt + (1/(2/len(uk)))*(self.analytical_flux(u) - self.analytical_flux(np.concatenate((np.array([u[-1]]),u[:-1]))))
        assert np.allclose(discretized_model(u), np.zeros_like(u)), f"Oh noooo! {np.linalg.norm(discretized_model(u))}"
        print("Yepppii!")
        
dt    = 0.01
numx  = 1638400
u0_func = lambda x: 2  + np.sin(np.pi * x)

model   = BurgerNewtonForm(dt, numx, u0_func)
uk = 0.5*( np.concatenate(( np.array([model._u0[-1]]) , model._u0[:-1])) +  model._u0)

preconditioner     = MultiGrid( model )
solver             = NewtonsMethod( model, preconditioner)

preconditioner.MGmaxlvl= 14
preconditioner.CFL     = 0.9
solver.max_ForcingTerm = 1e-10

P_uk1 = solver.solve(uk, model._u0, precondition=True)
model.check_solution(P_uk1)

# P_GMRES_it   = solver.history["GMRES_iterations"]
# P_GMRES_norm = solver.history["GMRES_minNorm"]
# P_n_Newton_it  = len(P_GMRES_it)
# P_forcing_terms = solver.history["forcing_terms"]
# solver.reset()

# for i in range(P_n_Newton_it):
#     if i == 0:        
#         plt.plot(np.log(P_GMRES_norm[i]),'-*' ,color="green",linewidth = 0.8, markersize=3, label = f"max d*t = {(preconditioner.CFL/max(abs(model._u0)))  * ((2/numx)/dt):3}")
#     else:
#         plt.plot(np.log(P_GMRES_norm[i]),'-*' ,color="green",linewidth = 0.8, markersize=3)

#     x_end = len(P_GMRES_norm[i]) - 1
#     y_end = np.log(P_GMRES_norm[i][-1])

#     #plt.text(x_end - 0.3, y_end, f"Newton_it = {i+1}", ha="right", va="center",fontsize=8)
#     plt.text(x_end - 0.3, y_end, f"TOL: {P_forcing_terms[i]:.1e}", ha="right", va="center",fontsize=4)

# plt.title(f"GMRES convergence plot: n = {numx}, dt = {dt}", fontsize=20)
# plt.xlabel("GMRES iterates", fontsize=10)
# plt.ylabel("log of error (2-norm)", fontsize=10)
# plt.legend()

# # results_dir = "/home/krane/dune-env/NUMN28/Project/Results"
# # os.makedirs(results_dir, exist_ok=True)

# # filename = os.path.join(results_dir, f"MeshConvergence_dt{dt}_numx{numx}_CFL{preconditioner.CFL}.png")
# # plt.savefig(filename, dpi=300, bbox_inches="tight")

# plt.show()


# lvls       = [2, 4, 6, 8]
# colors     = ["orange", "crimson", "dodgerblue", "darkviolet"]

#dts        = [0.1, 0.9, 1.2, 1.9]
#colors     = ["blue", "orange", "magenta", "limegreen"]

# numx_list  = [200,600]
# colors     = ["navy", "skyblue", "firebrick", "salmon"]
# labels     = ["Preconditioned n = 200", "Preconditioned n = 600", "Standard n = 200", "Standard n = 600"]
# P_GMRES_norm_list = []
# P_n_Newton_it_list = []
# P_forcing_terms_list = []

# for numx in numx_list:
    
#     model   = BurgerNewtonForm(dt, numx, u0_func)

#     uk = 0.5*( np.concatenate(( np.array([model._u0[-1]]) , model._u0[:-1])) +  model._u0)

#     preconditioner   = MultiGrid( model )
#     solver           = NewtonsMethod( model, preconditioner)
    
#     solver.max_ForcingTerm = 1e-10

#     P_uk1 = solver.solve(uk, model._u0, precondition=True)

#     P_GMRES_it   = solver.history["GMRES_iterations"]
#     P_GMRES_norm = solver.history["GMRES_minNorm"]
#     P_n_Newton_it  = len(P_GMRES_it)
#     P_forcing_terms = solver.history["forcing_terms"]
#     solver.reset()
    
#     P_GMRES_norm_list.append(P_GMRES_norm)
#     P_n_Newton_it_list.append(P_n_Newton_it)
#     P_forcing_terms_list.append(P_forcing_terms)



# for numx in numx_list:
    
#     model   = BurgerNewtonForm(dt, numx, u0_func)

#     uk = 0.5*( np.concatenate(( np.array([model._u0[-1]]) , model._u0[:-1])) +  model._u0)

#     preconditioner   = MultiGrid( model )
#     solver           = NewtonsMethod( model, preconditioner)
    
#     solver.max_ForcingTerm = 1e-10

#     P_uk1 = solver.solve(uk, model._u0, precondition=False)

#     P_GMRES_it   = solver.history["GMRES_iterations"]
#     P_GMRES_norm = solver.history["GMRES_minNorm"]
#     P_n_Newton_it  = len(P_GMRES_it)
#     P_forcing_terms = solver.history["forcing_terms"]
#     solver.reset()
    
#     P_GMRES_norm_list.append(P_GMRES_norm)
#     P_n_Newton_it_list.append(P_n_Newton_it)
#     P_forcing_terms_list.append(P_forcing_terms)
    

# "-----------Plotting-----------"
# plt.figure(figsize=(11,7))

# for j in range(len(colors)):
#     P_GMRES_norm = P_GMRES_norm_list[j]
#     P_forcing_terms = P_forcing_terms_list[j]
#     P_n_Newton_it = P_n_Newton_it_list[j]
    
#     for i in range(P_n_Newton_it):
#         if i == 0:        
#             plt.plot(np.log(P_GMRES_norm[i]),'-*' ,color=colors[j],linewidth = 0.8, markersize=3, label=labels[j])
#         else:
#             plt.plot(np.log(P_GMRES_norm[i]),'-*' ,color=colors[j],linewidth = 0.8, markersize=3)
    
#         x_end = len(P_GMRES_norm[i]) - 1
#         y_end = np.log(P_GMRES_norm[i][-1])
        
#         #plt.text(x_end - 0.3, y_end, f"Newton_it = {i+1}", ha="right", va="center",fontsize=8)
#         plt.text(x_end - 0.3, y_end, f"TOL: {P_forcing_terms[i]:.1e}", ha="right", va="center",fontsize=4)
        
# plt.title(f"GMRES convergence plot comparison: dt = {dt}", fontsize=20)
# plt.xlabel("GMRES iterates", fontsize=10)
# plt.ylabel("log of error (2-norm)", fontsize=10)
# plt.legend()

# results_dir = "/home/krane/dune-env/NUMN28/Project/Results"
# os.makedirs(results_dir, exist_ok=True)

# filename = os.path.join(results_dir, f"comparisonGMRES_dt{dt}_Forcing{solver.max_ForcingTerm:.1e}.png")
# plt.savefig(filename, dpi=300, bbox_inches="tight")

# plt.show()