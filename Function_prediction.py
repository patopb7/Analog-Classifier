import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, random, debug, vmap
import time
from utilfuncs import Mesher, computeLocalElements, computeFilter, get_dof#, plot_mesh, plot_and_export_mesh
from mmaOptimize import optimize
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder


def run_prediction(rho, indx):
    # Carga el dataset de iris
    iris = load_iris()
    X_raw = iris['data']
    y_raw = iris['target'].reshape(-1, 1)
    
    # Normalización y cambio de signo, escalado por 9.8
    X = -X_raw / jnp.max(X_raw)
    
    # Codificación one-hot
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_labels = encoder.fit_transform(y_raw)
    
    
    
    Xtrain = X
    Xtrian = jnp.array( Xtrain )
    ytrain = one_hot_labels
    ytrian = jnp.array( ytrain )
    

    
    data = {'Xtest':Xtrain, 'ytest':ytrain}
    
    
    
    #Mesh 
    nelx, nely = 80, 30
    elemSize = np.array([1., 1.])
    mesh = {'nelx':nelx, 'nely':nely, 'elemSize':elemSize,\
            'ndof':2*(nelx+1)*(nely+1), 'numElems':nelx*nely}
        
    #Material
    material = {'Emax':1., 'Emin':1e-3, 'nu':0.3, 'penal':3.}
    
    #Filter
    filterRadius = 2.0
    H, Hs = computeFilter(mesh, filterRadius)
    ft = {'type':1, 'H':H, 'Hs':Hs}
    
    #Boundary condition
    
    node_lr = 2*(nelx*(nely+1)+0)+1
    
    
    F0   = np.zeros((mesh['ndof'],1))
    
    dofs  = np.arange(mesh['ndof'])
    
    fixed = np.array(get_dof(0, nely, nely) + get_dof(nelx, nely, nely))
    free  = jnp.setdiff1d(np.arange(mesh['ndof']),fixed)
    
    print(fixed)
    
    load = [get_dof(20, 0, nely)[1], get_dof(30, 0, nely)[1], get_dof(50, 0, nely)[1], get_dof(60, 0, nely)[1] ]
    
    print(load)
    
    Utarget = [get_dof(30, 30, nely)[1], get_dof(40, 30, nely)[1], get_dof(50, 30, nely)[1]]
    
    print(Utarget)
    
    symXAxis = False
    symYAxis = False
    
    bc = {'fixed':fixed,'free':free,\
              'symXAxis':symXAxis, 'symYAxis':symYAxis,\
                  'Utarget':Utarget, 'load':load}
    
    print(['Utarget ', bc['Utarget']])    
    #Constrain
    globalVolumeConstraint = {'isOn':True, 'vf':0.7}
    
    #Optimize
    optimizationParams = {'maxIters':100,'minIters':10,'relTol':0.1}
    projection = {'isOn':False, 'beta':4, 'c0':0.5}
    
    
    
    print(Xtrain[0])
    
        
    
        
    #%%
    
    
    class ComplianceMinimizer:
        def __init__(self, mesh, bc, material, \
                     globalvolCons, projection, data, indx):
            self.mesh = mesh
            self.material = material
            self.bc = bc
            M = Mesher()
            self.edofMat, self.idx = M.getMeshStructure(mesh)
            self.K0 = M.getK0(self.material)
            self.globalVolumeConstraint = globalvolCons
            
            #self.consHandle = self.computeConstraints
            self.numConstraints = 1
            self.projection = projection
            self.data = data
            self.ltest  = len(data['ytest'])
            
            self.disp = jnp.array(bc['Utarget'])
            self.load = jnp.array(bc['load'])
            
            self.indx = indx
            
        #-----------------------#
    
        
        def computeDisplacement(self, rho):
            #-----------------------#
            @jit
            # Code snippet 2.9
            def projectionFilter(rho):
                if(self.projection['isOn']):
                    v1 = np.tanh(self.projection['c0']*self.projection['beta'])
                    nm = v1 + jnp.tanh(self.projection['beta']*(rho-self.projection['c0']))
                    dnm = v1 + jnp.tanh(self.projection['beta']*(1.-self.projection['c0']))
                    return nm/dnm
                else:
                    return rho
            #-----------------------#
            @jit
            # Code snippet 2.2
            def materialModel(rho):
                E = self.material['Emin'] + \
                    (self.material['Emax']-self.material['Emin'])*\
                                    (rho)**self.material['penal']
                return E
            #-----------------------#
            # @jit
            # #Code snippet 2.8
            # def materialModel(rho): # RAMP
            #     S = 8. # RAMP param
            #     E = 0.001*self.material['Emax'] +\
            #             self.material['Emax']*(rho/ (1.+S*(1.-rho)) )
            #     return E
            # Y = materialModel(rho)
            #-----------------------#
            @jit
            # Code snippet 2.3
            def assembleK(E):
                K_asm = jnp.zeros((self.mesh['ndof'], self.mesh['ndof']))
                K_elem = (self.K0.flatten()[np.newaxis]).T 
                # print(K_elem.shape)
    
                K_elem = (K_elem*E).T.flatten()
                K_asm = K_asm.at[(self.idx)].add(K_elem) #UPDATED
                return K_asm
            #-----------------------#
            @jit
            # Code snippet 2.4
            def solveKuf(K,Fuerza): 
                u_free = jax.scipy.linalg.solve\
                        (K[self.bc['free'],:][:,self.bc['free']], \
                        Fuerza[self.bc['free']], \
                         check_finite=False)
                u = jnp.zeros((self.mesh['ndof']))
                u = u.at[self.bc['free']].set(u_free.reshape(-1)) #UPDATED
                return u
            #-----------------------#
            @jit
            def Ucalculation(K,indx,Xtest,ytest): 
                F0 = jnp.zeros(self.mesh['ndof'])
                F0 = F0.at[self.load].set(Xtest[indx])
                
                u   = solveKuf(K,F0)
                u3  = u[self.disp]
                
    
                return u3
            #-----------------------#
            rho = projectionFilter(rho)
            E = materialModel(rho)
            K = assembleK(E)
            

    
            Uresponse = Ucalculation(K, self.indx, self.data['Xtest'], self.data['ytest'])
            
            
            
            
            
              
            return Uresponse
        #-----------------------#
    
    
    
    Opt = ComplianceMinimizer(mesh, bc, material, \
                    globalVolumeConstraint, projection, data, indx)
    
    
    #%%
    rho = np.load("rho_opt0.npy")
    U_prediction = Opt.computeDisplacement(rho)
    return U_prediction
