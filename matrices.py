import numpy as np
from scipy.linalg import block_diag
from warnings import warn

def heat_flow_function(nu, n):
    '''
    input:  nu: a scalar 
            n: an integer
    output: the heat flow matrix with dimension n**2 x n**2
    '''
    if nu<=0:
        warn("The heat flow matrix wont be positive definite")
    
    vec_nu=(-nu)*np.ones(n**2-n)
    vec_a=np.ones(n-1)*(-nu)
    vec_b=np.ones(n)*(1+4*nu)
    D=np.diag(vec_a, -1) + np.diag(vec_b) + np.diag(vec_a, 1)
    return block_diag(*([D] * n))+np.diag(vec_nu, n)+np.diag(vec_nu,-n)

def Pei_function(alpha, n):
    '''
    input:  alpha: a scalar
            n: an integer
    output: the Pei matrix with dimension n x n
    '''
    Pei_matrix=alpha*np.eye(n)+np.ones((n,n))
    return Pei_matrix

def VFH_function(K):
    '''
    input: K: the number of fractal iterations we want to use to define our matrix 
    
    output: the VFH matrix of dimension 5**k x 5**k
    '''

    H=np.diag(-2*np.ones(5))
    H[0]=np.ones(5)
    H[:,0]=np.ones(5)
    H[0,0]=-4

    p_11=3
    p_21=2
    p_31=5
    p_41=4

    p_1k_1=p_11
    p_2k_1=p_21
    p_3k_1=p_31
    p_4k_1=p_41

    if K<=0:
        raise ValueError('k has to be strictly positive')
    
    for k in range(2,K+1):
        #we define H_i depending on H_{i-1}
        #we already defined H_1 before the loop
        size=H.shape[0]
    
        if k>2:
            p_1k=5**(k-2)*(p_11-1)+p_1k_1
            p_2k=5**(k-2)*(p_21-1)+p_2k_1
            p_3k=5**(k-2)*(p_31-1)+p_3k_1
            p_4k=5**(k-2)*(p_41-1)+p_4k_1
        else:
            p_1k=p_11
            p_2k=p_21
            p_3k=p_31
            p_4k=p_41
    
        #putting -1 in the index to compensate for the indexing going from 0 to size-1 in numpy
        V_1=np.zeros((size, size))
        V_1[p_1k-1,p_2k-1]=1
        V_2=np.zeros((size, size))
        V_2[p_2k-1, p_1k-1]=1
        V_3=np.zeros((size, size))
        V_3[p_3k-1, p_4k-1]=1
        V_4=np.zeros((size, size))
        V_4[p_4k-1, p_3k-1]=1
    
        p_1k_1=p_1k
        p_2k_1=p_2k
        p_3k_1=p_3k
        p_4k_1=p_4k
    
        #Since we have a lot of block matrices, easier to explicitely complete it
        H_k=block_diag(*([H] * 5))
        H_k[size:2*size, 0:size]=V_1.T
        H_k[2*size:3*size, 0:size]=V_2.T
        H_k[3*size:4*size, 0:size]=V_3.T
        H_k[4*size:5*size, 0:size]=V_4.T
        H_k[0:size, size:2*size]=V_1
        H_k[0:size, 2*size:3*size]=V_2
        H_k[0:size, 3*size:4*size]=V_3
        H_k[0:size, 4*size:5*size]=V_4
    
        H=H_k
    return H

#ref: https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT3350/h07/undervisningsmateriale/chap9slides.pdf
def Poisson_function(k):
    '''
    input: k: an integer
    output: The poisson matrix of dimension k**2 x k**2
    '''

    I_k=np.eye(k)

    T=np.diag(2*np.ones(k))+np.diag(-1*np.ones(k-1), -1) + np.diag(-1*np.ones(k-1), 1)

    Poisson_matrix=np.kron(T,I_k)+np.kron(I_k, T)
    
    return Poisson_matrix



#source: https://people.sc.fsu.edu/~jburkardt/m_src/wathen/wathen.html  The wathen_ge.m file
def wathen_ge(nx, ny): 
    '''
    input: nx: an integer
           ny: an integer
    output: the wathen matrix of size (3*nx*ny+2*nx+2*ny+1) x (3*nx*ny+2*nx+2*ny+1)      
    '''
    n=3*nx*ny+2*nx+2*ny+1
    A=np.zeros((n,n))

    EM=np.array([
     [6.0, -6.0,  2.0, -8.0,  3.0, -8.0,  2.0, -6.0],
    [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
     [2.0, -6.0,  6.0, -6.0,  2.0, -8.0,  3.0, -8.0],
    [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
     [3.0, -8.0,  2.0, -6.0,  6.0, -6.0,  2.0, -8.0],
    [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
     [2.0, -8.0,  3.0, -8.0,  2.0, -6.0,  6.0, -6.0],
    [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0] ])

    node = np.zeros(8)

    for j in range(1, ny+1):
        for i in range(1, nx+1):
#For the element (I,J), determine the indices of the 8 nodes.

            node[0] = ( 3 * j     ) * nx + 2 * j + 2 * i + 1
            node[1] = node[0] - 1
            node[2] = node[0] - 2

            node[3] = ( 3 * j - 1 ) * nx + 2 * j + i - 1
            node[7] = node[3] + 1

            node[4] = ( 3 * j - 3 ) * nx + 2 * j + 2 * i - 3
            node[5] = node[4] + 1
            node[6] = node[4] + 2

            rho = np.random.uniform(0,50)

            for krow in range(0, 8):
                for kcol in range(0, 8):
                    A[int(node[krow])-1,int(node[kcol])-1]=A[int(node[krow]-1),int(node[kcol])-1]+ rho * EM[krow,kcol]

    return A

def Lehmer_function(k):
    '''
    Outputs the Lehmer matrix of dimension k x k for a given k 
    
    intput: k: integer, the size of the matrix
    output: A: The Lehmer matrix of dimension k x k 
    '''
    
    A=np.zeros((k,k))
    
    for i in range(k):
        for j in range(k):
            if j>=i:
                A[i,j]=(i+1)/(j+1) #need to compensate with +1 because of the way numpy indexes the matrices
                A[j,i]=A[i,j]
    return A 




