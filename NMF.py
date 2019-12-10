"""

@author: Ming Sheng Choo
@E-mail: cming0721@gmail.com
@Github: MingSheng92

Code reimplemented based on psuedo code presented in the research paper.

"""
import numpy as np

def nmfPG(V, Winit, Hinit, tol, maxiter):
    # W,H: output solution
    # Winit,Hinit: initial solution
    # tol: tolerance for a relative stopping condition
    # timelimit, maxiter: limit of time and iterations
    W = Winit.copy()
    H = Hinit.copy()
    
    # gradient for W and H, used for projection optimization later
    gradW = np.dot(W, np.dot(H, H.T)) - np.dot(V, H.T)
    gradH = np.dot(np.dot(W.T, W), H) - np.dot(W.T, V)
    
    # calculate initial gradient 
    initgrad = np.linalg.norm(np.r_[gradW, gradH.T])
    
    #print('Init gradient norm ', initgrad)
    #print("\n")
    
    tolW = max(1e-3, tol) * initgrad
    tolH = tolW 
    
    for iter in range(maxiter):
        # stopping condition
        projnorm = objective(H, W, gradH, gradW)
        # check for convergance
        if projnorm < tol * initgrad:
            break
        
        # Update basis and mixture matrix.
        W, gradW, iterW = nls_subproblem(V.T, H.T, W.T, tolW, 1000)
        W = W.T
        gradW = gradW.T
        
        if iterW == 0:
            tolW = 0.1 * tolW
            
        H, gradH, iterH = nls_subproblem(V, W, H, tolH, 1000)
        if iterH == 0:
            tolH = 0.1 * tolH
        
    # return W and H 
    return W, H    
	
def nls_subproblem(V,W,Hinit,tol,maxiter):
    # H, grad: output solution and gradient
    # iter: #iterations used
    # V, W: constant matrices
    # Hinit: initial solution
    # tol: stopping tolerance
    # maxiter: limit of iterations
    H = Hinit.copy()
    WtV = np.dot(W.T, V)
    WtW = np.dot(W.T, W)
    
    alpha = 1
    beta = 0.1 
    sub_iter = 20
    
    for iter in range(maxiter):
        grad = np.dot(WtW, H) - WtV
        projgrad = np.linalg.norm(grad[np.logical_or(grad < 0, H >0)])
        if projgrad < tol:
            break
        # search for step size alpha
        for n_iter in range(sub_iter):
            #Hn = max(H - alpha * grad, 0)
            Hn = H - alpha*grad
            Hn = np.where(Hn > 0, Hn, 0)
            d = Hn - H
            gradd = np.multiply(grad, d).sum()
            dQd = np.multiply(np.dot(WtW, d), d).sum()
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            
            if n_iter == 0:
                decr_alpha = not suff_decr
                Hp = H
            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha *= beta
            else:
                if not suff_decr or (Hp == Hn).all():
                    H = Hp
                    break
                else:
                    alpha /= beta
                    Hp = Hn

    return H, grad, iter
        
def objective(H, W, gH, gW):
    #Compute projected gradients norm.
    return np.linalg.norm(np.r_[gW[np.logical_or(gW<0, W>0)],
                                 gH[np.logical_or(gH<0, H>0)]])