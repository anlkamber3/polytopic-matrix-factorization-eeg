import numpy as np 
from utils import *
from tqdm import tqdm

def P_L1NormBall(Hi):
    H=Hi.T
    Hshape=H.shape
    lr=np.tile(np.reshape((1/np.linspace(1,Hshape[1],Hshape[1])),(1,Hshape[1])),(Hshape[0],1))

    u=-np.sort(-np.abs(H),axis=1)
    sv=np.cumsum(u,axis=1)
    q=np.where(u>((sv-1)*lr),np.tile(np.reshape((np.linspace(1,Hshape[1],Hshape[1])-1),(1,Hshape[1])),(Hshape[0],1)),np.zeros((Hshape[0],Hshape[1])))
    rho=np.max(q,axis=1)
    rho=rho.astype(int)
    lindex=np.linspace(1,Hshape[0],Hshape[0])-1
    lindex=lindex.astype(int)

    theta=np.maximum(0,np.reshape((sv[tuple([lindex,rho])]-1)/(rho+1),(Hshape[0],1)))
    ww=np.abs(H)-theta
    H=np.sign(H)*(ww>0)*ww
    return H.T

def P_NNL1NormBall(X):
    return P_L1NormBall(X * (X>0))


def P_LInftyNormBall(X):
    return X*(X>=-1.0)*(X<=1.0)+(X>1.0)*1.0-1.0*(X<-1.0)


def P_NNLInftyNormBall(X):
    return X*(X>0)*(X<=1.0)+1.0*(X>1)

def PMF(raw, X, r = 40, NumberofIterations = 2801, muv=10**(9.), epsv=1e-9, tuning=False, tune=300,tmin = -0.01,tmax = 2.5, threshold = 0.25):
    #This is our data to separate
    # X =raw[:][0]
    M,N = X.shape

    #Syntetic Data for variance normalization
    seed_num = 431
    np.random.seed(seed_num)

    S = P_L1NormBall(2*np.random.rand(r,N)-1.0)
    H = np.random.randn(M,r)
    X_syn = np.dot(H,S) + 0.00000001*np.random.randn(M,N)
    targetvar = np.var(X_syn)
    varX = np.var(X,1)

    D=np.diag(np.sqrt(targetvar/varX))

    X = np.dot(D,X) #Variance Normalization of X

    # ALGORTITHM STATE INITIALIZATION
    Z = (np.random.rand(r,N) - 0.5)/4 

    RX=np.dot(X,X.T)/N
    RXinv=np.linalg.pinv(RX+epsv*np.eye(M))

    tuning_epochs = []
    tuning_rvs = []

    for k in tqdm(range(NumberofIterations)):

        RcZ = np.dot(Z,Z.T)/N
        RZ = RcZ + epsv*np.eye(r)
        RZinv = np.linalg.inv(RZ)
        RZX = Z @ X.T / N
        Hhat = RZX.T @ RZinv
        RE = RX - Hhat @ RZX
        REinv = np.linalg.inv(RE + epsv * np.eye(M))
        
        DelZ = Hhat.T @ REinv @ (X- Hhat @ Z)
        Zu = Z + muv * DelZ * (0.99**k) / (np.linalg.norm(DelZ)+1e-10)
           
        if np.mod(k,50)==0:
            Z = P_L1NormBall(Zu)
        else:
            Z = Zu

    #Epoch tuning
        if (tuning == True) and (((np.mod(k,tune) == 0) and k != 0) or k == NumberofIterations-1) :

            # WZ = X
            W = X @ np.linalg.pinv(Z)

            #REVERT THE VARIANCE NORM
            
            W = np.linalg.inv(D) @ W

            rvs, gofs = calculate_rvs(raw,W,M,N,r,tmin,tmax)
            rv_count = sum(1 for rv in rvs if rv < threshold)
            tuning_epochs.append(k)
            tuning_rvs.append(rv_count)


    # WZ = X
    W = X @ np.linalg.pinv(Z)

    #REVERT THE VARIANCE NORM
    
    W = np.linalg.inv(D) @ W

    if tuning == True:
        return W, Z, tuning_epochs, tuning_rvs
    else:
        return W, Z