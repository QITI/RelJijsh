import numpy as np
import torch

def chain(N):
	"""Given a number of ions N the function produces a chain lattice in the
	form of a numpy array"""
	row_indices, col_indices = np.triu_indices(N,1)
	i1=np.triu_indices(N,1)
	i2=np.triu_indices(N,2)
	all_ones=np.zeros((N,N))
	all_ones[i1]=1
	all_ones2=np.zeros((N,N))
	all_ones2[i2]=1
	chain_lattice=all_ones-all_ones2
	chain_lattice=chain_lattice[row_indices,col_indices]
	return chain_lattice
	
def circle(N):
	"""Given a number of ions N the function produces a circle lattice in the
	form of a numpy array"""
	chain_lattice=chain(N)
	chain_lattice[N-2]=1
	return chain_lattice

def torch_normalized(x):
	"""Returns a torch vector x that is normalized by dividing by L2-norm"""
	norm = torch.sqrt((x ** 2).sum(0))
	return x/norm
	
def np_normalized(x):
	"""Returns a numpy vector x that is normalized by dividing by L2-norm"""
	norm = np.linalg.norm(x)
	return x/norm
	
def noise_adder(v,var,size):
	"""Adds Gaussian Noise to a vector,v, with variance, var, and produces a
	a specified number these noisy vectors all contained in a matrix."""
	noisy_vectors = np.zeros((size,len(v)))
	for i in range(len(vector)):
		noisy_vectors[:,i] = np.random.normal(v[i],var,size)
	return noisy_vectors