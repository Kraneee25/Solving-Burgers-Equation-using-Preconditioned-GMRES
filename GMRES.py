# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 21:09:32 2025
@author: krane
"""
from KrylovSpace import KrylovSpace
from numpy.linalg import norm
from scipy.linalg import solve_triangular

class GMRES(KrylovSpace):
    
    def __init__(self, A, x0, b, right_preconditioner= None, 
                 left_preconditioner=None):
        
        self.guess = x0
        self.GMRESproblem = A
        self.rhs = b
        self.GMRES_minNorms = []
        self.GMRES_iters = 0
        self.failures = 0
        
        if right_preconditioner is not None:
            self.GMRESproblem = lambda x: A( right_preconditioner(x) )

        if left_preconditioner is not None:
            self.GMRESproblem = lambda x: left_preconditioner( A(x) )
            self.rhs = left_preconditioner(b)
            
        self.r0 = self.rhs - self.GMRESproblem(x0)
        
        super().__init__(self.GMRESproblem,self.r0)
        
    def solve(self, TOL, max_iterations = 5000):
        
        for _ in range(max_iterations):
            
            QT , R_full = super().decomposeHessenberg()
            gVector = QT.T[:,0]*norm(super().residualGuess, ord=2)
            minNorm = abs(gVector[-1])
            
            self.GMRES_minNorms.append(minNorm)
            self.GMRES_iters += 1
            
            if _%10 == 0:
                print(f"\r   GMRES STATUS: Iteration = {_:6}  ||  norm_i = {minNorm}")
                
            if  minNorm <= TOL:
                y = solve_triangular(R_full[:-1], gVector[:-1])
                Vk = super().space[:,:-1]
                
                return self.guess + Vk@y
            
            try:            
                super().extendKrylovSpace()
                
            except:
                self.failures += 1
                print("No solution that satisfied the condition was found in the fully spanned Krylov Space")
                print(f"Minimum error norm for GMRES: {minNorm}")

                y = solve_triangular(R_full[:-1], gVector[:-1])
                Vk = super().space[:,:-1]
                
                return self.guess + Vk@y
            
            self.failures += 1
        raise Exception(f"The algorithm did not converge to a solution after {max_iterations} iterations.")

        
        
        