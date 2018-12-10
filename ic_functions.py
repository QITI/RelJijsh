import numpy as np
import torch

def chain(n):
	row_indices, col_indices = np.triu_indices(n,1)
	i1=np.triu_indices(n,1)
	i2=np.triu_indices(n,2)
	all_ones=np.zeros((n,n))
	all_ones[i1]=1
	all_ones2=np.zeros((n,n))
	all_ones2[i2]=1
	chain_lattice=all_ones-all_ones2
	chain_lattice=chain_lattice[row_indices,col_indices]
	return chain_lattice
	
def circle(n):
	chain_lattice=chain(n)
	chain_lattice[n-2]=1
	return chain_lattice

