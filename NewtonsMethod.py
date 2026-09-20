# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:21:44 2025
@author: krane
"""

from numpy import  allclose, zeros_like
from numpy.linalg import norm as norm
from GMRES import GMRES

class NewtonsMethod(GMRES):
    """ Solves Newton's Method using GMRES """ 
    def __init__(self, model, MGpreconditioner):
        
        self.zeroProblem = model.zeroProblem
        self.MG = MGpreconditioner
        self.inexactJacobian = model.inexactJacobian_operator
        self.history = {"forcing_terms": [],
                        "GMRES_iterations": [],
                        "GMRES_minNorm": [],
                        "NEWTON_norm": []}
        self.MGmaxlvl = 2
        self.max_ForcingTerm = 0.9
            
    def reset (self):
        "For multiple runs, we do not need to create a diff class instance"
        self.history = {"forcing_terms": [],
                        "GMRES_iterations": [],
                        "GMRES_minNorm": [],
                        "NEWTON_norm": []}
        self.MGmaxlvl = 2
        self.max_ForcingTerm = 0.9
        
    def get_ForcingTerm (self, currentIterate, oldIterate, TOL, 
                         oldForcingTerm, gamma=0.1, max_ForcingTerm=0.9):
        """ Eisenstat and Walker 's method to define the forcing terms.
        Main issue is this determines the accuracy of GMRES. Meaning, 
        if the forcing term is too high, GMRES will not be able to solve
        for a more accurate solution, i.e. Newtons method may not terminate"""
        
        if allclose(oldForcingTerm, 0.0):
            return max_ForcingTerm
        
        norm2CurrentEval = norm( self.zeroProblem(currentIterate),ord=2 )
        norm2OldEval = norm( self.zeroProblem(oldIterate),ord=2 )
        ForcingTermA = gamma*(norm2CurrentEval/norm2OldEval)**2
        
        if gamma*oldForcingTerm**2 <= 0.1:
            ForcingTerm = min(max_ForcingTerm, ForcingTermA) 
            
        else:
            forcingtermB = max( ForcingTermA, gamma*oldForcingTerm**2 )
            ForcingTerm = min( max_ForcingTerm, forcingtermB )
        
        forcingtermD = max( ForcingTerm, 
                           0.5*TOL/norm(self.zeroProblem(currentIterate)))
        return min(  max_ForcingTerm,  forcingtermD )
     
    def solve(self, guess, prev, rtol=1e-10, atol=0, max_iterations=10000, 
              precondition = False):
        
        termination_critertia = rtol*norm( self.zeroProblem(guess),ord=2) + atol
        print(f"NEWTON STATUS: Iteration = {0:5} --- {norm(self.zeroProblem(guess),ord=2):.9f}")
        
        ForcingTerm = self.max_ForcingTerm
        
        #statistics
        self.history["forcing_terms"].append(ForcingTerm)
        self.history["NEWTON_norm"].append(norm(self.zeroProblem(guess),ord=2))

        du = zeros_like(guess)
        uk = guess.copy()
       
        A   = self.inexactJacobian(uk)
        b   = -self.zeroProblem(uk) 

        if precondition:
            
            def preconditioner (b):
                return  self.MG.computeMG(zeros_like(b), b, ul=uk, 
                                          l = self.MGmaxlvl)      
            
            super().__init__(A, du, b, right_preconditioner = preconditioner)
            e_du = super().solve(TOL=ForcingTerm)  
            
            #statistics
            self.history["GMRES_iterations"].append(self.GMRES_iters)
            self.history["GMRES_minNorm"].append(self.GMRES_minNorms)
            
            du = preconditioner(e_du)
            
        else :
            
            super().__init__(A, du, b)
            du = super().solve(TOL=ForcingTerm)  
            
            #statistics
            self.history["GMRES_iterations"].append(self.GMRES_iters)
            self.history["GMRES_minNorm"].append(self.GMRES_minNorms)
                
        uk1 = uk + du

        for i in range(max_iterations):
            
            self.history["NEWTON_norm"].append(norm(self.zeroProblem(uk1),ord=2))
            if norm( self.zeroProblem(uk1),ord=2) <= termination_critertia:
                print(f"\nThe algorithm converged after {i} iterations with the 2-norm error: {norm( self.zeroProblem(uk1),ord=2)}")
                        
                return uk1
            
            print(f"NEWTON STATUS: Iteration = {i+1:5} --- {norm(self.zeroProblem(uk1),ord=2):.9f}")            

            ForcingTerm = self.get_ForcingTerm(uk1, uk, termination_critertia, 
                                               ForcingTerm,  
                                               max_ForcingTerm=self.max_ForcingTerm)
            
            #statistics
            self.history["forcing_terms"].append(ForcingTerm)
            
            du = zeros_like(uk)
            uk   = uk1.copy()
            
            A = self.inexactJacobian(uk)
            b = -self.zeroProblem(uk)
            
            if precondition:
                
                def preconditioner (b):
                    return  self.MG.computeMG(zeros_like(b), b, ul=uk, 
                                              l = self.MGmaxlvl)  
                
                super().__init__(A, du, b, right_preconditioner = preconditioner)
                e_du = super().solve(TOL=ForcingTerm)
                
                #statistics
                self.history["GMRES_iterations"].append(self.GMRES_iters)
                self.history["GMRES_minNorm"].append(self.GMRES_minNorms)

                du = preconditioner(e_du)
           
            else :
                
                super().__init__(A, du, b)
                du = super().solve(TOL=ForcingTerm)    
                
                #statistics
                self.history["GMRES_iterations"].append(self.GMRES_iters)
                self.history["GMRES_minNorm"].append(self.GMRES_minNorms)
            
            uk1 = uk + du
            
        raise Exception(f"The algorithm did not converge to a solution after {max_iterations} iterations.")
