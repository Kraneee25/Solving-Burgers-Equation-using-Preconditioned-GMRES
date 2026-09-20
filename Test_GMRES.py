# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:11:08 2025
@author: krane
"""

from numpy import allclose
from scipy.linalg import solve
from  numpy.random import rand 
from GMRES import GMRES

"""---FOR TESTING GMRES.py---"""

problem = rand(100,100)
rightHandSide = rand(100)
solution = solve(problem, rightHandSide)

Test = GMRES(problem, solution-1e2 , rightHandSide)
approximation = Test.solve(1e-9)

def test_Solver():
    assert allclose(approximation,solution)