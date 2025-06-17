#!/usr/bin/env python

import os, sys, traceback
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as pylab
import numpy as np
import pylab as pl
import scipy as sci
import warnings

from cvxopt import matrix, spmatrix, solvers, printing
solvers.options['show_progress'] = False

### ------------------------------- Optimal Transport ---------------------------------------

########### Compute transport with a LP Solver

def computeTransportLP(distribWeightS,distribWeightT, distances):
	# init data
	Nini = len(distribWeightS)
	Nfin = len(distribWeightT)
	

	# generate probability distribution of each class
	p1p2 = np.concatenate((distribWeightS,distribWeightT))
	p1p2 = p1p2[0:-1]
	# generate cost matrix
	costMatrix = distances.flatten()

	# express the constraints matrix
	I = []
	J = []
	for i in range(Nini):
		for j in range(Nfin):
			I.append(i)
			J.append(i*Nfin+j)
	for i in range(Nfin-1):
		for j in range(Nini):
			I.append(i+Nini)
			J.append(j*Nfin+i)

	A = spmatrix(1.0,I,J)

	# positivity condition
	G = spmatrix(-1.0,range(Nini*Nfin),range(Nini*Nfin))

	sol = solvers.lp(matrix(costMatrix),G,matrix(np.zeros(Nini*Nfin)),A,matrix(p1p2))
	S = np.array(sol['x'])

	Gamma = np.reshape([l[0] for l in S],(Nini,Nfin))
	return Gamma

########### Compute transport with the Sinkhorn algorithm
## ref "Sinkhorn distances: Lightspeed computation of Optimal Transport", NIPS 2013, Marco Cuturi

def computeTransportSinkhorn(distribS,distribT, M, reg):
	# init data
	Nini = len(distribS)
	Nfin = len(distribT)
	
	numItermax = 200
	cpt = 0
	
	# we assume that no distances are null except those of the diagonal of distances
	u = np.ones(Nini)/Nini
	uprev=np.zeros(Nini)
	
	K = np.exp(-reg*M)
	Kp = np.dot(np.diag(1/distribS),K)
	transp = K
	cpt = 0
	err=1
	while (err>1e-4 and cpt<numItermax):
		if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
# 			print 'Infinity'
			if cpt!=0:
				u = uprev
			break
		uprev = u
		v = np.divide(distribT,np.dot(K.T,u))
		u = 1./np.dot(Kp,v)
		if cpt%10==0:
			# we can speed up the process by checking for the error only all the 20th iterations
			transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
			err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
		cpt = cpt +1


	return np.dot(np.diag(u),np.dot(K,np.diag(v)))

def computeTransportSinkhorn1(distributS, distributT, M, reg, numItermax=1000, stopThr=1e-9):
    dim_S = len(distributS)
    dim_T = len(distributT)
    
    # 归一化成本矩阵以提高数值稳定性
    M = M - np.max(M, axis=1, keepdims=True)
    K = np.exp(-M / reg)
    Kp = np.dot(np.diag(1 / distributS), K)
    
    u = np.ones(dim_S) / dim_S
    v = np.ones(dim_T) / dim_T
    
    err = 1
    for i in range(numItermax):
        uprev = u
        vprev = v
        
        KtransposeU = np.dot(K.T, u)
        v = distributT / (KtransposeU + 1e-8)  # 避免除零
        u = 1. / (np.dot(Kp, v) + 1e-8)  # 避免除零
        
        if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isinf(u)) or np.any(np.isinf(v)):
            warnings.warn(f'Numerical errors at iteration {i}')
            u = uprev
            v = vprev
            break
        
        if i % 10 == 0:
            transp = np.dot(np.diag(u), np.dot(K, np.diag(v)))
            err = np.linalg.norm(np.sum(transp, axis=0) - distributT)**2
            if err < stopThr:
                break
    
    return np.dot(np.diag(u), np.dot(K, np.diag(v)))

def diracize(M):
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if fabs(M[i][j])>0:
				M[i][j]=0
			else:
				M[i][j]=1

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def computeTransportSinkhornLabels(distribS,LabelsS, distribT, M, reg):
	viz = False
	
	# init data
	Nini = len(distribS)
	Nfin = len(distribT)
	
	numItermax = 2000
	cpt = 0
	
	indices_labels = []
	num_label = len(np.unique(LabelsS))
	idx_begin = np.min(LabelsS)
	for c in range(idx_begin,num_label+1):
		idxc = indices(LabelsS, lambda x: x==c)
		indices_labels.append(idxc)

	# we assume that no distances are null except those of the diagonal of distances
	u = np.ones(Nini)/Nini
	uprev=np.zeros(Nini)
	
	
	K = np.exp(-reg*M)
	Kp = np.dot(np.diag(1/distribS),K)

	while (sci.linalg.norm(u-uprev,1)>1e-6 and cpt<numItermax):
		if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			if cpt!=0:
				u = uprev
			break
		uprev = u
		u = 1/np.dot(Kp,distribT/np.dot(K.T,u))
		cpt = cpt +1
	v = distribT/np.dot(K.T,u)

	transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
	#print 'nb iter= '+str(cpt)

	# the transport has been computed. Check if classes are really separated
	allnorms=[]
	for t in range(Nfin):
		column = transp[:,t]
		norms=[]
		for c in range(num_label):
			norms.append(np.linalg.norm(column[indices_labels[c]]))
		allnorms.append(norms)

	for t in range(Nfin):
		# which is the class which has the greatest norm ?
		c = np.argmax(allnorms[t])
		transp[indices_labels[c],t]=transp[indices_labels[c],t]/distribT[t]
		for otherlabel in range(num_label):
			if otherlabel!=c:
				transp[indices_labels[otherlabel],t]=0



	return transp




def computeTransportSinkhornLabelsLpL1(distribS,LabelsS, distribT, M, reg, eta=0.1):
	viz = False
	p=1./2.
	epsilon=1e-3

	# init data
	Nini = len(distribS)
	Nfin = len(distribT)
	
	numItermax = 100
	#print distribT
	
	indices_labels = []
	idx_begin = np.min(LabelsS)
	for c in range(idx_begin,np.max(LabelsS)+1):
		idxc = indices(LabelsS, lambda x: x==c)
		indices_labels.append(idxc)
	transp = []

	#print LabelsS
	W=np.zeros(M.shape)

	for cpt in range(10):
		# we assume that no distances are null except those of the diagonal of distances
		u = np.ones(Nini)/Nini
		v = np.ones(Nfin)/Nfin
		uprev=np.zeros(Nini)
		
		Mreg = M + eta*W
		
		K = np.exp(-reg*Mreg)
		Kp = np.dot(np.diag(1/distribS),K)
		
		transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))

		cpt = 0
		err=1
		while (err>1e-4 and cpt<numItermax):
			if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
				# we have reached the machine precision
				# come back to previous solution and quit loop
# 				print 'Infinity'
				if cpt!=0:
					u = uprev
				break
			uprev = u
			v = np.divide(distribT,np.dot(K.T,u))
			u = 1./np.dot(Kp,v)
			if cpt%10==0:
				# we can speed up the process by checking for the error only all the 20th iterations
				transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
				err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
			cpt = cpt +1
		# the transport has been computed. Check if classes are really separated
		W = np.ones((Nini,Nfin))
		for t in range(Nfin):
			column = transp[:,t]
			for c in range(len(indices_labels)):
				col_c = column[indices_labels[c]]
				W[indices_labels[c],t]=(p*((sum(col_c)+epsilon)**(p-1)))

	
	return transp

