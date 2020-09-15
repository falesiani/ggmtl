#        <Graph Guided Multi-Task Learning>
# 	  
#   File:     GGMTL.py
#   Authors:  Francesco Alesiani <Francesco.Alesiani@neclab.eu>
#             
# 
# NEC Laboratories Europe GmbH, Copyright (c) <2020>, All rights reserved.  
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  
#        PROPRIETARY INFORMATION ---  
# 
# SOFTWARE LICENSE AGREEMENT
# 
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
# 
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
# 
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor. 
# 
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
# 
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
# 
# COPYRIGHT: The Software is owned by Licensor.  
# 
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
# 
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
# 
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
# 
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
# 
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
# 
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
# 
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.  
# 
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
# 
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
# 
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
# 
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.  
# 
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
# 
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
# 
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
# 
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
# 
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
# 
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
# 
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from mykNN import mykNN

from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import csr_matrix, bsr_matrix, coo_matrix, block_diag
from scipy import sparse
from scipy.io import loadmat

import warnings
try:
    from CCMTL import CCMTL
except:
    warnings.warn("CCMTL package not present, performance may not be as expected")

        

def mtliter2(func):
    def wrapper_mtliter(x,y, *args, **kwargs):
        return [ func(_x,_y,*args, **kwargs) for _x,_y  in zip(x,y) ]
    return wrapper_mtliter
def mtliter3(func):
    def wrapper_mtliter(x,y, *args, **kwargs):
        return np.mean([ func(_x,_y,*args, **kwargs) for _x,_y  in zip(x,y) ])
    return wrapper_mtliter
@mtliter2
def mtl_rmse(yp,y):
    return ((yp.ravel()-y.ravel())**2).mean()**.5
@mtliter3
def mtl_mrmse(yp,y):
    return ((yp.ravel()-y.ravel())**2).mean()**.5
def get_var_mat(mat_data,names):
    vars_ = []
    for name in names:
        vars_.append(mat_data[name])
    return vars_
def fit_start( X, Y, edges=None, rho=10, k=10, metric='euclidean', verbose=True,add_const=False, use_solve = False):
        task_num = len(X)
        dimension = X[0].shape[1] + (1 if add_const else 0)
        if edges is None: alpha = 1e+6
        else:             alpha = 1      
        XX = []
        sparseX = []
        W = np.zeros([task_num, dimension]) 
        C = np.zeros([task_num, dimension])
        for i in range(task_num):
            if add_const:
                Xi = np.ones([X[i].shape[0], dimension])
                Xi[:, :-1] = X[i]
            else:
                Xi = X[i]        
            yi = Y[i]
            XXi = Xi.T.dot(Xi)
            XYi = Xi.T.dot(yi)            
            if use_solve: 
                wi = np.linalg.solve(XXi+alpha*np.eye(dimension), XYi)   
            else:
                wi = np.linalg.pinv(XXi).dot(XYi)    
                
            W[i] = wi.ravel()           
            C[i] = XYi.ravel()
            sparseX.append(coo_matrix(Xi))
            XX.append(coo_matrix(XXi))

        A = block_diag(XX)
        X = block_diag(sparseX)
     
        C = C.ravel()
        Y = np.concatenate(Y, axis=0)
        Y = Y.reshape([-1,1])
        C = C.reshape([-1,1])        
        if edges is None:
            edges = mykNN(W, k)
        return W,edges,X,Y,A,C
    
def get_variables( X, Y, add_const=False):
        task_num = len(X)
        if add_const:
            dimension = X[0].shape[1] + 1
        else:
            dimension = X[0].shape[1]
        XX = []
        sparseX = []
        XY = np.zeros([task_num, dimension])
        for i in range(task_num):
            if add_const:
                Xi = np.ones([X[i].shape[0], dimension])
                Xi[:, :-1] = X[i]
            else:
                Xi = X[i]
            yi = Y[i]
            XXi = Xi.T.dot(Xi)
            XYi = Xi.T.dot(yi)
            XY[i] = XYi.ravel()
            sparseX.append(coo_matrix(Xi))
            XX.append(coo_matrix(XXi))
        XX = block_diag(XX)
        X = block_diag(sparseX)
        XY = XY.ravel()
        Y = np.concatenate(Y, axis=0)        
        Y = Y.reshape([-1,1])
        XY = XY.reshape([-1,1])
        return X,Y,XX,XY    

def get_variable_X( X, add_const=False):
        task_num = len(X)
        if add_const:
            dimension = X[0].shape[1] + 1
        else:
            dimension = X[0].shape[1]
        sparseX = []
        for i in range(task_num):
            if add_const:
                Xi = np.ones([X[i].shape[0], dimension])
                Xi[:, :-1] = X[i]
            else:
                Xi = X[i]
            sparseX.append(coo_matrix(Xi))
        X = block_diag(sparseX)
        return X   

def get_laplacian_edges_v2(edges,K):
    Le = np.sum([v*get_bij(i,j,K).toarray() for (i,j,v) in edges],axis=0)
    return Le
def get_laplacian_edges_v3(edges,lij,K):
    Le = np.sum([v*lij[kij]*get_bij(i,j,K).toarray() for kij,(i,j,v) in enumerate(edges)],axis=0)
    return Le
def get_bij(i,j,K):
    return coo_matrix(([1,1,-1,-1], ([i,j,i,j], [i,j,j,i])), shape=(K,K))
def get_B(edges,K):
    B = np.concatenate([get_bij(i,j,K).toarray() for (i,j,v) in edges],axis=1)
    B = coo_matrix(B)
    return B
def evaluate_rmse(V,X,Y):
    trainerr = X.dot(V.reshape([-1,1]))-Y
    trainerr = trainerr.ravel()
    msetrain = ((trainerr*trainerr).mean())**.5
    return msetrain

def predict_vec(X,V):
    return X.dot(V.reshape([-1,1]))

def compute_zeros(edges):
    ee = np.array([_[-1] for _ in edges])
    return len(ee[np.where(ee==0)])/len(edges)*100.

def get_lij(W,edges):
    if type(edges) == list:
        edges = np.array(edges)
    i = np.array(edges[:, 0], dtype=int)
    j = np.array(edges[:, 1], dtype=int)
    lpq = 0.5 / (np.sqrt(np.sum((W[i]-W[j])**2, axis=1)) + 1e-6)
    return lpq
def get_hyper_gradient(edges,Le,XX,XY,X,Y,Vprec,Xval,Yval,rho,K,dim,alpha=1e-6, add_const=False):
    m = len(edges)
    Id = np.eye(dim+(1 if add_const else 0))
    Im = np.eye(m)
    A = (rho*np.kron(Le,Id)+XX.toarray())
    A = A + alpha*np.eye(*A.shape)
    
    A = coo_matrix(A)
    XY = coo_matrix(XY)
    if A.shape[0]  < 100:
        V = spsolve(A, XY)
    else:
        V  = cg(A, XY.A, tol=1e-6, maxiter=10000)[0] 
    trainerr = X.dot(V.reshape([-1,1]))-Y
    msetrain = (trainerr**2).mean()**.5
    B = get_B(edges,K)
    Vp = sparse.kron(V.T,Im)
    Bp = sparse.kron(B.T,Id)
    valerr = Xval.dot(V.reshape([-1,1]))-Yval
    mseval = (valerr*valerr).mean()**.5
    Vval = Xval.T.dot(valerr)
    Vval = coo_matrix(Vval)
    if A.shape[0] < 100:
        Vtemp = spsolve(A, Vval)
    else:
        Vtemp  = cg(A, Vval.A, tol=1e-6, maxiter=10000)[0] #x0=W_vec,  
    dde = Vp.dot(Bp).dot(Vtemp).reshape([-1,1])    
    return dde,V,mseval,msetrain
def update_edges(edges,dde,mu,vmin=0.,vmax=1e6):
    return [(i,j,min(vmax,max(vmin,v-mu*float(dde[kij])))) for kij,(i,j,v) in enumerate(edges)]
def get_adjacent(edges,undirected=True):
    if type(edges)==list:
        edges = np.array(edges)
    i = np.array(edges[:, 0], dtype=int)
    j = np.array(edges[:, 1], dtype=int)
    eij = edges[:, 2] 
    if undirected:
        ij = np.concatenate([i,j],axis=0)
        ji = np.concatenate([j,i],axis=0)
        eij = np.concatenate([eij,eij],axis=0)
        i = ij
        j = ji
    n_nodes = max(i.max(),j.max())+1    
    A = coo_matrix((eij, (i, j)), shape=(n_nodes, n_nodes))
    return A
###################
from sklearn.model_selection import train_test_split
def train_test_split_mtl(Xalls,yalls, test_size=0.4, random_state=0):
    X_trains, X_tests, y_trains, y_tests = [],[],[],[]
    for _X,_y in zip(Xalls,yalls):
        X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=test_size, random_state=random_state)
        X_trains+= [X_train]
        X_tests+=[X_test]
        y_trains+=[y_train]
        y_tests+=[y_test]
    return X_trains, X_tests, y_trains, y_tests

def set_edges_default(edges,default=1.):
    return [(i,j,default) for kij,(i,j,v) in enumerate(edges)]


def post_process(X_trains, y_trains, edges, rho, verbose, model_class):
    model_sparse = model_class()
    edgesp = [_ for _ in edges if _[-1]!=0]
    if len(edgesp)==0:
        edges = None
        if verbose: print('graph with no edges')
        print('WARNING: graph with no edges:: results may not be accurate...')
    else:
        edges = np.array(edgesp)
    model_sparse.fit( X_trains, y_trains, edges = edges, k=None, rho=rho, verbose=verbose);
    return model_sparse

class GGMTL:
    def __init__(self, tol=1e-6):
        self.tol = tol
        self.verbose = False
        self.add_const = False
        self.use_solve = False
        self.model_sparse = None
    def rmse(self, X_tests, y_tests):
        y_predicts = self.predict(X_tests)
        return np.mean(mtl_rmse(y_predicts,y_tests))
    def mse(self, X_tests, y_tests):
        y_predicts = self.predict(X_tests)
        return np.mean(mtl_mse(y_predicts,y_tests))      
    def predict(self,X_tests):
        if self.model_sparse:
            y_predicts = self.model_sparse.predict(X_tests)
        else:
            Xtst = get_variable_X(X_tests,add_const=self.add_const)
            y_predicts = predict_vec(Xtst,self.V)
        return y_predicts    
    def post_process(self, X_trains, y_trains, rho, verbose, model_class):
        model_sparse =  post_process(X_trains, y_trains, self.edges, rho, verbose, model_class )
        return model_sparse       
    def fit(self,X_trains,y_trains,niters=10,xi=1,eta=1,gamma=10,k=10, beta = 1e-3, norm_l21_flag = True,rho = 1e4,mu=-1e-4,vmin=0,vmax=1,verbose=True,do_split_flag=False,alpha=1e-6,lambdav=1.,edges = None,default_edge_value=0., test_size=0.2
        , rho_mtl = None,model_class=None ):
        eps = 1e-6
        if not edges is None:
            assert(type(edges)==list and len(edges[0])==3), "edges need to be a list of tripe (i,j,v)"
        As = [] #the list of the adjacent matrices, the first is the kNN
        #split train and validation
        X_trs, X_vals, y_trs, y_vals = train_test_split_mtl(X_trains,y_trains, test_size=test_size, random_state=0)
        K = len(X_trains)
        self.num_task = K
        dim = X_trains[0].shape[1]
        Xval,Yval,XXval,XYval = get_variables(X_vals, y_vals,add_const=self.add_const)
        if edges is None:
            V,edges,Xtr,Ytr,XXtr,XYtr = fit_start(X_trs, y_trs, k=k,add_const=self.add_const)
        else:
            V,edges_,Xtr,Ytr,XXtr,XYtr = fit_start(X_trs, y_trs, k=k,add_const=self.add_const)
        As.append(get_adjacent(edges))
        msetr = evaluate_rmse(V,Xtr,Ytr)
        if verbose: plt.plot([_[-1] for _ in edges],'b<')
        if verbose: print("ITL msetr={}".format(msetr))
        edges = set_edges_default(edges,default_edge_value)
        Le =  get_laplacian_edges_v2(edges,K)
        mseval_prec,msetrain_prec= 1e8,1.2e8
        for _ in range(niters):
            if do_split_flag:
                X_trs, X_vals, y_trs, y_vals = train_test_split_mtl(X_trains,y_trains, test_size=test_size)
                Xval,Yval,XXval,XYval = get_variables(X_vals, y_vals,add_const=self.add_const)
                Xtr,Ytr,XXtr,XYtr = get_variables(X_trs, y_trs,add_const=self.add_const)
            if norm_l21_flag:
                W = V.reshape(K,dim+(1 if self.add_const else 0))
                lij = get_lij(W,edges)
                Le =  get_laplacian_edges_v3(edges,lij,K)
            else:
                Le =  get_laplacian_edges_v2(edges,K)

            dde,V,mseval,msetrain = get_hyper_gradient(edges,Le,XXtr,XYtr,Xtr,Ytr,V,Xval,Yval,rho,K,dim,alpha,add_const=self.add_const)

            noise = np.random.randn(len(edges),1)*beta
            e = np.array([_[-1] for _ in edges]).reshape([-1,1])
            dde = lambdav*dde - xi*e - eta*np.sign(e) + gamma*np.log(eps+e) + noise   
            edges = update_edges(edges,dde,mu=mu,vmin=vmin,vmax=vmax)
            mu=mu*.9
            As.append(get_adjacent(edges))
            if verbose: plt.plot([_[-1] for _ in edges],'r^')
            if verbose: 
                _ = mseval,msetrain,compute_zeros(edges),np.mean(np.array(edges)[:,-1]),np.mean(dde)
                print("mseval={:.5f},msetr={:.5f},nz(edges)={:.1f}, mean(edges)={:.5f}, mean(dde)={:.5f}".format(*_))
            change = abs((mseval-mseval_prec)/mseval_prec) + abs((msetrain-msetrain_prec)/msetrain_prec)
            mseval_prec,msetrain_prec=mseval,msetrain
            if change<self.tol:
                break                
        if verbose: 
            plt.plot([_[-1] for _ in edges],'k>')
            plt.show(block=False)
        
        self.edges,self.V,self.As = edges,V,As
        if not model_class is None:
            self.model_sparse = self.post_process(X_trains, y_trains, rho=rho_mtl, verbose=verbose, model_class = model_class)
        return edges,V,As


###################

from itertools import  product
def hyper_opt_grid_search(X_trains, y_trains,X_tests,y_tests
                          , niterss,xis,etas,gammas,rhos,betas,norm_l21_flags, mus,rhos_mtl
                          , model_class, k=10, verbose=False, default_edge_value=.0):
    results = []
    for hps in product(niterss,xis,etas,gammas,rhos,betas,norm_l21_flags, mus,rhos_mtl):
        (niters,xi,eta,gamma,rho,beta,norm_l21_flag, mu,rho_mtl) = hps
        ggmtl = GGMTL()
        edges,V,As = ggmtl.fit(X_trains
                               , y_trains
                               , niters = niters
                               , xi = xi
                               , eta = eta
                               , gamma = gamma
                               , k = 10
                               , beta = beta
                               , norm_l21_flag = norm_l21_flag
                               , rho = rho
                               , mu=mu
                               , vmin=0
                               , vmax=1
                               , verbose=verbose
                               , do_split_flag=False
                               , default_edge_value=default_edge_value
                              , model_class = model_class
                              , rho_mtl = rho_mtl)
        rmse_post= mtl_mrmse(ggmtl.predict(X_tests),y_tests)
        rmse = ggmtl.rmse(X_tests, y_tests)
        nz = compute_zeros(ggmtl.edges)
        nedges = len(ggmtl.edges)
        titles_ = ['niters','xi','eta','gamma','rho','beta','norm_l21_flag', 'mu', 'rmse_post','rmse','nz','ggmtl','model_sparse','nedges','default_edge_value']
        if model_class is None:
            model_sparse = ggmtl
        else:
            model_sparse = ggmtl.model_sparse
        data_ = [niters,xi,eta,gamma,rho,beta,norm_l21_flag, mu, rmse_post,rmse,nz,ggmtl,model_sparse,nedges,default_edge_value]
        results+= [{k:v for k,v in zip(titles_,data_)}]
        best_conf = min(results, key=(lambda k: k['rmse_post']))
    return results, best_conf

def mtl_hyper_parameter_search(X_trains, X_tests, y_trains, y_tests, mtl_model , rhos = [1e-4,1e-3,1e-2,.1, 1,10,100, 1e3, 1e4, 1e5 ], verbose=False):
    results_=[]
    for rho in rhos:
        model_ = mtl_model()
        model_.fit( X_trains, y_trains, k=10, rho=rho, verbose=False);
        y_predicts = model_.predict(X_tests)
        rmse_= np.mean(mtl_rmse(y_predicts,y_tests))
        if verbose:  print(rho,rmse_)
        results_+=[{'rho':rho,'rmse':rmse_,'model':model_}]
    best_conf_ = min(results_, key=(lambda k: k['rmse']))
    model=best_conf_['model']
    rho_mtl = best_conf_['rho']
    rmse_mtl = best_conf_['rmse']
    if verbose: print('mtl rho = {}, rmse = {:.3}'.format(rho_mtl,rmse_mtl ))
    return model,rho_mtl,rmse_mtl,best_conf_,results_


# comparison with MTL
def compare_performances(X_tests,y_tests,model,model_sparse, verbose=False):
    def perc(a,b):
        return (a-b)/a*100.
    rmse_before,rmse_after =  np.mean(mtl_rmse(model.predict(X_tests),y_tests)), np.mean(mtl_rmse(model_sparse.predict(X_tests),y_tests))
    ne_after,ne_before = len(model_sparse.edges), len(model.edges)
    rmse_perc = perc(rmse_before,rmse_after) 
    nz_perc = perc(ne_after,ne_before)
    if verbose: print('compare rmse, nz (\%)= ', rmse_perc,nz_perc)
    if verbose: print('compare rmse, nz = ',rmse_before, rmse_after, ne_before, ne_after)
    return rmse_perc, nz_perc, rmse_before, rmse_after, ne_before, ne_after

def read_dataset(filename):
    mat_data = loadmat(filename)
    X_trains, X_tests, y_trains, y_tests =  get_var_mat(mat_data,['X_train','X_test','Y_train','Y_test'])
    return X_trains, X_tests, y_trains, y_tests
def generate_dataset_iter(dsnames,dataset_global,path):
    for dataset_name in dsnames:
        yield dataset_global,dataset_name,path + dataset_name +".mat"
def get_coords(filename, coord_name = 'w_corrdinates'):
    mat_data = loadmat(filename)
    if 'w_corrdinates' in mat_data:
        return  get_var_mat(mat_data,[coord_name])[0]
    else:
        return None

###################

# conda install -c bioconda mcl
# pip install markov_clustering[drawing]
import markov_clustering as mc
def complete_cluster(clusters,nodes):
    node_in = [i for c in clusters for i in c ]
    node_out = [i for i in nodes if not i in node_in]
    if len(node_out)>0:
        clusters += [tuple(node_out)]
    return clusters

def cluster_plot(adjacent, title, pos, inflation=1.3,filename=None, labels=None, label_flag=True, node_size=150,figsize=(6,6),use_nodeaslabel=False,width=1):
    fig, ax = plt.subplots(1,1, sharey=False, sharex=False,figsize=figsize)    
    result = mc.run_mcl(adjacent, inflation=inflation)          
    clusters = mc.get_clusters(result)    # get clusters
    plt.title(title)    
    graph = nx.Graph(adjacent)
    clusters = complete_cluster(clusters,graph.nodes())
    if pos is None:
        pos=nx.spring_layout(graph,iterations=100)    
    mc.draw_graph(adjacent, clusters, pos=pos, with_labels=False, edge_color="silver",node_size=node_size,width=width)  
    if labels is None:
        if use_nodeaslabel:
            labels = {n: n for ni,n in enumerate(graph.nodes())}
        else:
            labels = {n: ci for ci,c in enumerate(clusters) for n in c}        
    if pos is None:
        pos=nx.spring_layout(graph,iterations=100)
    if label_flag:
        nx.draw_networkx_labels(graph, pos, labels=labels )
    if not filename is None: 
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show(block=False)
    return clusters, pos, labels

def plot_clustered_graphs(As, dataset_name, inflation=1.01, pos_fix = None
                          ,label_flag=True, use_final_pos_flag=True
                          , use_first_flag=True, show_intermediate_flag=False, knn_different_flag = True,figsize=(8,8),use_nodeaslabel=False,width=1,node_size=150):
    filename_fn = lambda ds,ki: "{}_{}_{}_labels.png".format(ds,ki, 'with' if label_flag else 'no')  
    pos = None
    if use_final_pos_flag and pos_fix is None:
        graph = nx.Graph(As[-1].A)
        pos_final=nx.spring_layout(graph,iterations=100)  
    if not pos_fix is None:
        pos_final = pos_fix
        pos = pos_fix
    for ki,a in enumerate(As):
        if not show_intermediate_flag: filename= None
        if ki>1 and use_final_pos_flag:
            pos = pos_final
        if ki>0 and not use_final_pos_flag:
            pos = pos_cluster            
        is_first = False
        is_last = False
        if ki==0:
            title = "kNN"
            is_first = True
            filename = filename_fn(dataset_name,"kNN")
            if knn_different_flag:
                pos = None
        elif ki==len(As)-1:
            is_last = True
            title = "GGMTL"
            filename = filename_fn(dataset_name,"GGMTL")
        else:
            title="{}-iter".format(ki)
        if show_intermediate_flag or is_first or is_last:
            clusters, pos_cluster, labels = cluster_plot(a.A, title=title, pos=pos, inflation=inflation,filename=filename,label_flag=label_flag,figsize=figsize,use_nodeaslabel=use_nodeaslabel,width=width,node_size=node_size)
            plt.show(block=False)
            
            
            
###################