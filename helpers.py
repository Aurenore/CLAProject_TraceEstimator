import numpy as np 
from math import fabs
from warnings import warn
import time
import matplotlib.pyplot as plt

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

def algorithm_1(A, u, function, maxit=50, epsilon=1e-5):
    # to do: save T_j as sparse
    '''
    Implements algorithm 1 from Bai et al. It computes a lower/upper bound of the quantity u^T f(A) u by using the Gauss-Radau rules
    INPUT:
    - A: a symmetric positive definite matrix of size n times n for some n, with eigenvalues in [a,b]
    - u: vector of size n 
    - f: smooth function in the interval [a,b]
    - maxit: maximum number of iteration
    - epsilon: tolerance between two iterations
    
    OUTPUT:
    - [U,L]: Upper and Lower bound of the quantity u^T f(A) u by using the Gauss-Radau rule.
    '''
    # Remark 1: compute a and b 
    interval = Gerschgorin(A)

    if interval[0]<=0:
        interval[0] = 1e-4
    #print("a=", interval[0])  Armelle: I've put these two as comments
    #print("b=", interval[1])
    if (np.linalg.eigvals(A)<=0).any():
        warn("The matrix A should be positive definite.")
        print("eigenvalues of A:", np.linalg.eigvals(A))
    if not (A==A.T).all():
        warn("A is not symmetric. Please choose A such that A=A.T")
        print("A =",A)

    # set the first variables
    x_j1 = u/np.linalg.norm(u)    
    gamma_j1 = 0.0
    I_j = np.zeros(2)
    I_j1 = np.zeros(2)

    # stopping criteria for I_j^U and I_j^L
    above_thresh_mask = np.array([True, True]) 
    indices = np.arange(2)

    # Save X as a matrix (for re-orthogonalization)
    n=len(u) 
    X=np.zeros((n, maxit+1))
    X[:,0] = x_j1

    # iteration
    for j in range(maxit):
        w = A@X[:,j]
        alpha_j=X[:,j].T@w
        r_j = w - alpha_j*X[:,j]
        if j>0:
            r_j = r_j - gamma_j1*X[:,j-1]
        
        #reorthogonlization to avoid roundoff error encured by Lanczos
        alphas=X.T@r_j
        r_j=r_j-X@alphas
        alpha_j=alpha_j+alphas[-1]
        gamma_j = np.linalg.norm(r_j)

        # build T_j:
        if j==0:
            T_j = np.array([alpha_j])
        else:
            # horizontal array [0, ..., 0, gamma_{j-1}]
            temp_h = np.expand_dims(np.zeros(T_j.shape[0]),1)
            temp_h[-1] = gamma_j1
            # vertical array [0, ..., 0, gamma_{j-1}, alpha_j].T
            temp_v = np.expand_dims(np.zeros(T_j.shape[0] + 1),1)
            temp_v[-1] = alpha_j
            temp_v[-2] = gamma_j1
            # new T_j:
            T_j = np.hstack((np.vstack((T_j, temp_h.T)), temp_v))
        
        # for Gauss Radau, a or b have to be zeros of the polynomial, i.e. must be eigenvalues of T_tilda_{j'}:
        for i in indices[above_thresh_mask]: # for lower and upper bounds
            T_tilda_j = T_tilda(T_j, gamma_j, interval[i])
            
            # compute eigenvalues of T_tilda_j:            
            theta_k, eig_vec = np.linalg.eigh(T_tilda_j)
            w_k_square = eig_vec[0, :]*eig_vec[0,:]
            I_j[i] = function(theta_k).dot(w_k_square)

            # stopping criterion
            if (j>0) & (np.abs(I_j[i] - I_j1[i]) <= epsilon*np.abs(I_j[i])):
                above_thresh_mask[i] = False
                
        # check stopping criterion  
        if not above_thresh_mask.any(): #when both are false: break.
            break
        
        x_j1 = r_j/gamma_j
        X[:,j+1]=x_j1.copy()
        I_j1 = I_j.copy()
        gamma_j1 = gamma_j.copy()
        
        if not check_ortho(X[:,0:j+1]):
            warn("The algorithm does not build an orthonormal basis, at j ="+str(j))

    return u.dot(u)*I_j   

def algorithm_2(A, m, p, function, epsilon):
    '''
    A: a symmetric positive definite matrix of size n times n for some n  
    m: the chosen number of samples
    p: the chosen probabibilty for the confidence interval, the trace of function(A) will be in the bound with proba
       AT LEAST p 
    
    return: I: an unbiased estimator of the quantity tr(f(A))
            (L_p, U_p): the bounds of the confidence interval of tr(f(A)) with proba p 
    '''
    
    if np.any(A!=A.T):
        raise ValueError('The matrix isnt symmetric')
        
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError('The matrix isnt positive definite')
        
    if p<=0 or p>=1:
        raise ValueError('we dont have 0<p<1')
        
    if m>A.shape[0]/2: #arbitrary choice
        print('Notice that for the use of the Montecarlo method to be justified, its better to have m<=A.shape[0]/2')
        
    n=A.shape[0]
    L=np.zeros(m) 
    U=np.zeros(m) 
    L_min=10^8 #dummy value to start with for L_min
    U_max=-10^8 #dummy value to start with, for U_max
    
    
    for j in range(m):
        z = np.random.uniform(0,1,n)
        z[z<0.5]=-1
        z[z>=0.5]=1
        #apply algorithm 1 to obtain a lower bound L_j and an upper bound U_j and set 
        
        #algorithm 1 will output z.T@function(A)@z and we know from the paper that this is an unbiased estimator of 
        #tr(function(A))
        L[j], U[j]=algorithm_1(A=A, u=z, function=function, maxit=200, epsilon=epsilon)
        L_j, U_j=L[j],U[j]
        
        #computing the mean as we have an unbiased estimator
        I=np.ones(m).T@(L+U)/(2*(j+1)) #need to add +1 because of indexization in python. The first vector multiplication
                                       #sums all the non-zero elements of L+U
        L_min=np.minimum(L_min,L_j)
        U_max=np.maximum(U_max,U_j)
        eta_2=(-0.5*(j+1)*(U_max-L_min)**2)*(np.log(1-p)/2)
        
        #computing the mean as we have an unbiased estimator
        L_p_j= np.ones(m).T@L/(j+1)-np.sqrt(eta_2)/(j+1)
        U_p_j= np.ones(m).T@U/(j+1)+np.sqrt(eta_2)/(j+1)
        
    return U_p_j, L_p_j, I  


def numerical_experiments(matrix):
    '''
    runs the numerical experiments and returns 2 array with the running time and the value found for each experiment
    
    matrix: numpy array the matrix we want to study
    '''
    #defining the required values
    
    n=matrix.shape[0]
    matrix_running_time=np.zeros(4)
    matrix_trace_value=np.zeros(4)
    tol = 1e-5
    
    def f(x):
        return 1/x
    
    #running time of algo 2

    start=time.time()
    L=algorithm_2(matrix,m=int(n/2),p=0.5, function=f, epsilon=tol) #for some m,p to tune 
    execution_algo_2=time.time()-start 

    matrix_running_time[0]=execution_algo_2
    matrix_trace_value[0]=L[2]
    
    #running time using built in numpy functions
    start=time.time()
    Tr_A_inv=np.trace(np.linalg.inv(matrix))
    execution_built_in=time.time()-start 
    matrix_running_time[1]=execution_built_in
    matrix_trace_value[1]=Tr_A_inv
    
    #running time using n linear systems

    start=time.time()
    Tr_A_inv=0

    for i in range (n):
        e=np.zeros(n)
        e[i]=1
        Tr_A_inv+=e.T@np.linalg.solve(matrix,e)

    execution_linear=time.time()-start
    matrix_running_time[2]=execution_linear
    matrix_trace_value[2]=Tr_A_inv
    
    #using algorithm 1
    start=time.time()
    Tr_A_inv=0

    for i in range (n):
        e=np.zeros(n)
        e[i]=1
        Tr_A_inv+=algorithm_1(matrix,e,function=f, maxit=50, epsilon=tol)

    execution_algo_1=time.time()-start
    matrix_running_time[3]=execution_algo_1
    matrix_trace_value[3]=(Tr_A_inv[0]+Tr_A_inv[1])/2
    
    return matrix_running_time, matrix_trace_value

def graph(matrix, Ms, savefile):
    '''
    Plot the exact solutions and the estimations given by algorithm 2, applied on the matrix given as argument. 

    matrix: numpy array the matrix we want to study
    Ms: list of the parameter 'm' we would like to study 
    '''
    #defining the required values
    
    n=matrix.shape[0]
    tol = 1e-5
    
    def f(x):
        return 1/x
    
    Tr_A_inv=np.trace(np.linalg.inv(matrix))
    U_p = np.zeros(len(Ms))
    L_p = np.zeros(len(Ms))
    I = np.zeros(len(Ms))

    for k, m in enumerate(Ms):
        U_p[k], L_p[k], I[k] = algorithm_2(matrix, m=m, p=0.95, function=f, epsilon=tol) 

    fig = plt.figure(figsize=(8,8))
    plt.plot(Ms, Tr_A_inv*np.ones(len(Ms)))
    plt.plot(Ms, U_p, '-.')
    plt.plot(Ms, L_p, '-.')
    plt.plot(Ms, I, '-x')
    plt.legend(['exact value', 'lower bound', 'upper bound', 'estimated value'])
    plt.xlabel('Number of samples')
    plt.ylabel('Trace')
    plt.title('Matrix of size ' + str(n) + 'x' + str(n)) 
    plt.savefig('figures/' + savefile +'.png')