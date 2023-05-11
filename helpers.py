import numpy as np 
from math import fabs

def T_tilda(T, gamma, a):
    """
    Implements the correction mentioned in Remark 3. 
    INPUT:
    - T: T_j
    - gamma: gamma_j
    - a: scalar, a or b
    OUTPUT:
    - T_tilda_j: matrix as defined page 76 of Bai et al., for the Gauss-Radau case.
    """
    e_j = np.zeros(T.shape[0])
    e_j[-1] = 1
    delta = np.linalg.solve(T-a*np.eye(T.shape[0]), gamma*gamma*e_j) # tridiagonal system, could be written without np.linalg.solve maybe
    phi = a + delta[-1]
    temp_left = np.vstack((T, gamma*e_j))
    temp_right = np.expand_dims(np.append(gamma*e_j, phi),1)
    return np.hstack((temp_left, temp_right))

def Gerschgorin(A):
    """Simple way to find a and b in R such that the eigenvalues of A are in [a,b].
    INPUT:
    - square matrix A
    OUTPUT:
    - array: [a,b]
    """
    for i in range(A.shape[0]):
        r = 0
        pivot = A[i,i]
        for j in range(A.shape[1]):
            if i!=j:
                r += fabs(A[i][j])

        if i==0:
            a = pivot-r
            b = pivot+r
        else:
            if (pivot-r)<a:
                a = pivot-r
            if (pivot+r)>b:
                b = pivot+r

    return np.array([a, b])    


def check_new_eig(a, eigs, tol=1e-12):
    """
    Checks if a is in array eigs.
    INPUT: 
    - a: number
    - eigs: eigenvalues
    - tol: tolerance 
    OUTPUT:
    - bool: true if a is in eigs, else false.
    """
    for i in range(len(eigs)):
        if np.abs(a-eigs[i])<tol:
            return True
    return False

        
def check_ortho(X, tol=1e-8):
    """
    Checks if the columns of X are orthogonal to one another.
    INPUT:
    - X: 2d matrix, its columns are the ones we want to check the orthogonality of
    OUTPUT:
    - bool: 
        - True: the vectors are orthogonal
        - False: they are not.
    """
    prod = X.T @ X
    for i in range(prod.shape[0]):
        for j in range(prod.shape[1]):
            if i==j:
                if np.abs(prod[i,j]-1)>tol:
                    return False
            else:
                if np.abs(prod[i,j])>tol:
                    return False
    return True