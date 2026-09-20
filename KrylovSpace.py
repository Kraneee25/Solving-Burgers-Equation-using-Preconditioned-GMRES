# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 19:03:57 2022

@author: Krane
"""

from numpy import array, hstack, zeros, vstack, dot, ndarray, eye, outer
from numpy.linalg import norm 

class KrylovSpace:
    "NOTE!: The matrices involved here are 'transposed' for convenience, i.e. columns in paper are rows in the code"
    "One can set a new guess by changing the variable residualGuess, this will also initiate a new space"
    def __init__(self, problem, residualGuess):
        
        "A can be a ndarray object, or a function"
        if isinstance(problem, ndarray) and problem.ndim == 2:
            self.__problem = lambda x: problem@x
        else:
            self.__problem = problem
        
        self._residualGuess = residualGuess
        
        self.initiateKrylovSpace()
        
    def initiateKrylovSpace (self):
        "The orthonormal space that spans the Krylov space (k+1) will be initiated"
        self.__space = array([self._residualGuess/norm(self._residualGuess,ord=2)])
        "The Hessenberg matrix will be initiated"
        H11 = dot(self.__problem(self.__space[0]),self.__space[0]) #(Av1,v1)
        H12 = norm(self.__problem(self.__space[0]) - H11*self.__space[0]) #||v2||
        self.__space = vstack((self.__space,(self.__problem(self.__space[0]) - H11*self.__space[0])/H12))
        
        self.__Hessenberg = array([[H11,H12]])
    
    @property
    def residualGuess(self):
        return self._residualGuess
    
    @property
    def space (self):
        return self.__space.T
    
    @property
    def Hessenberg(self):
        return self.__Hessenberg.T

    def extendHessenbergMatrix (self, newVector):
        "To extend, we first append a zero vector at the last row of the matrix"
        
        self.__Hessenberg = hstack(  (self.__Hessenberg, zeros( (self.__Hessenberg.shape[0], 1) ))  )
        self.__Hessenberg = vstack( (self.__Hessenberg, newVector) )
        
    def extendKrylovSpace (self): 
        "The vectors generated are orthonormal to the previous ones"
        
        if len(self.__space)>len(self._residualGuess):
            raise Exception("Krylov subspace is full: no additional independent vectors can be generated beyond the rank of A.")
            
        rawBasisVector = self.__problem(self.__space[-1])
        newBasisVector = rawBasisVector.copy()
        
        innerProducts = []
        
        for orthnBasisVector in self.__space:
            
            innerProduct = dot(newBasisVector,orthnBasisVector)
            innerProducts.append( innerProduct )
            newBasisVector -= innerProduct*orthnBasisVector
        
        normNextVector = norm( newBasisVector,ord=2 )
        
        if normNextVector < 1e-14:
            raise Exception("Happy breakdown")
    
        # if not allclose(normNextVector, 0, atol=1e-12):
        newVector = innerProducts + [normNextVector] 
        self.extendHessenbergMatrix( newVector )
    
        newBasisVector = newBasisVector/normNextVector
        self.__space = vstack( (self.__space,newBasisVector) )
            
    def decomposeHessenberg (self, TOL = 1e-8):
        "Decomposes the Hessenberg matrix to QR using Householder"
        
        triangMatrix = self.__Hessenberg.T.copy() #This gets modified under the iteration process.
        length, numVectors = triangMatrix.shape #length is the number of elements in a vector, i.e. num of rows.
        orthoMatrices = eye(length) #We initiate the Q's with an identity matrix.
        
        for i in range(numVectors):
            rawOrthoMatrix = eye(length)

            z = triangMatrix[i:,i]
            Pz = array([norm(z)] + [0]*(len(z)-1))
            v = Pz-z
        
            if norm(v, ord=2) > TOL:
                rawOrthoMatrix[i:,i:] = eye(len(z)) - 2*outer(v,v)/norm(v, ord=2)**2
                
            orthoMatrices = rawOrthoMatrix@orthoMatrices
            triangMatrix = rawOrthoMatrix@triangMatrix
        

        return orthoMatrices.T, triangMatrix

