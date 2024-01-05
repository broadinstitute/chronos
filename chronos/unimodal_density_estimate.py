import numpy as np
from scipy.linalg import eig

'''
A python implementation of the Bernstein polynomial decomposition of a probability density with unimodal constraint,
as presented by Turnball and Ghosh, "Unimodal density estimation using Bernstein polynomials", Comput. Stat. Data Anal. (2014) 
https://doi.org/10.1016/j.csda.2013.10.021. This is a python adaptation of their provided R code.
'''

def m_opt_CN(t_data, L):
	'''get the optimal number of weights m'''
	n = len(t_data)
	m = np.floor(n**(1/3)) - 1

	logratio = 1

	while logratio < np.sqrt(n):
		m += 1
		B = np.stack([
			pbeta(t_data, shape1=k, shape2=m-k+1)
			for k in range(1:m+1)
		])
		Dmat = B.T.dot(L).dot(B)

		d = eig(Dmat)[0]
		min_eigenvalue = max(min(d), 0)
		max_eigenvalue = max(d)

		logratio = np.log10(max_eigenvalue) - np.log10(min_eigenvalue)

	return m - 1

def max_weight(m, F):
	'''Find the index of the maximum weight. To enforce unimodality of the function,
	the weights must also be unimodal.
	Parameters:
		`m` (`int`): number of weights/Bernstein polynomials
		`Fn` (function):
	Returns:
		int: the argmax
	'''
	return np.argmax( Fn(np.arange(1, m+1)/m) - Fn(np.arange(0, m)/m) )


def constraint_mat(m, maxplace):
	'''
	Function to generate the constraint matrix
	Parameters:
		`m` (`int`): number of weights/polynomials
		`maxplace` (`int`): the peak weight
	returns:
		`np.ndarray` containing the unimodal weight constraint matrix
	'''
	left_top_row = np.array([-1, 1] + [0]*(m-1))
	left_edge = np.stack([np.roll(left_top_row, i) for i in range(maxplace-1)])
	right_bottom_row = np.array([0]*(m-1) + [1, -1])
	right_edge = np.stack([np.roll(right_bottom_row, i) for i in range(-maxplace+2, 1)])
	return np.stack([
		np.ones(shape=m),
		np.diag(np.ones(shape=m)),
		left_edge,
		right_edge
	]).T

def solve_weights(m, Fn, lower, upper, Dmat, dvec):
	max_place = max_weight(m, Fn)
	Amat = constraint_mat(m, max_place)
	bvec = np.array([1] + [0]*(2*m-1))
