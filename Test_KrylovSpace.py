# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:11:08 2025
@author: krane
"""

from numpy import allclose, eye, tril, all
from  numpy.random import rand 
from KrylovSpace import KrylovSpace

"""---FOR TESTING KrylovSpace.py---"""

A = rand(100, 100)
b = rand(100)

Test = KrylovSpace(A, residualGuess=b)
for _ in range(len(b)-2):
    Test.extendKrylovSpace()

QT,R_full = Test.decomposeHessenberg()


def test_orthogonalization():
    assert allclose(Test.space@Test.space.T, eye(Test.space.shape[0]))
    
def test_Hessenberg():
    assert allclose(Test.Hessenberg, QT@R_full)

def test_triangular():
    tol=1e-10
    assert all(abs(tril(R_full[:-1], k=-1)) < tol)
    
    
    