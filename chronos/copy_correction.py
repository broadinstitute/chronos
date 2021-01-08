from patsy import dmatrix
import numpy as np
import pandas as pd
import tensorflow as tf


def get_shifts(gene_effect, copy_number):
	ge = gene_effect.copy()
	ge -= ge.median()
	cn = copy_number.loc[ge.index, ge.columns].fillna(1)[ge.notnull()].stack()
	return pd.DataFrame({
		"gene_effect_shift": ge.stack().values,
		"cn": cn.values,
		"cell_line_name": cn.index.get_level_values(0),
		"gene": cn.index.get_level_values(1)
		})
	
def logspace(low, high, n):
	start = 0
	end = np.log(high - low + 1)
	steps = np.linspace(start, end, n)
	converted = np.exp(steps) + low - 1
	return converted


def add_global_shift(cn, y, means, dtype, nknots_cn=10, nknots_ge=5, alpha=.2):
	np_dtype = {tf.double: np.double, tf.float32: np.float32}[dtype]
	knots_cn = list(cn.quantile(np.linspace(0, 1, nknots_cn)))
	knots_cn[0] += 1e-1
	knots_cn[-1] -= 1e-1

	knots_ge = list(logspace(np.quantile(means, 0.01), np.quantile(means, .99), nknots_ge))

	spline_gc = np.array(dmatrix(
				"te( \
					bs(cn, knots=%r, degree=3, include_intercept=False), \
					bs(means, knots=%r, degree=3, include_intercept=False) \
				)" % (knots_cn, knots_ge), 
				{"cn": cn.values, 'means': means}, return_type='matrix'
			))
	print('constructed spline matrix of shape %i, %i' % spline_gc.shape)
	_spline = tf.constant(spline_gc, dtype=dtype)
	_y = tf.constant(y.values, dtype=dtype)
	var = y.var()
	init = np.random.uniform(-.001, -.0001, size=(spline_gc.shape[1]))
	v_coeffs = tf.Variable(init.reshape((-1, 1)), dtype=dtype)
	v_weights = tf.Variable(1e-6 * np.ones(len(spline_gc)), dtype=dtype)
	_weights = tf.exp(-tf.abs(v_weights))
	_weight_cost = tf.reduce_mean(tf.square(v_weights))

	_out = _weights * tf.matmul(_spline, v_coeffs)[:, 0]	
	_cost = tf.reduce_mean(tf.square(_out - _y) )
	optimizer = tf.train.AdamOptimizer(.005)
	_step = optimizer.minimize(_cost + alpha * _weight_cost, var_list=[v_coeffs, v_weights])
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for i in range(501):
		sess.run(_step)
		if not i%100:
			print('\tcost:', sess.run(_cost))
	out = sess.run(_out)
	weights = sess.run(_weights)


	return weights, sess.run(_out)



def get_adjusted_matrix(shifts, gene_effect,):
	ge = gene_effect.stack()
	means = gene_effect.mean()

	adjusted = pd.Series(
		shifts['adjusted'].values + means.loc[shifts.gene].values,
		index=ge.index
	).reset_index()

	adjusted = pd.pivot(adjusted, index="level_0", columns="level_1")[0]
	adjusted.index.name = "cell_line_name"
	adjusted.columns.name = "gene"
	return adjusted



def alternate_CN(gene_effect, copy_number, nknots_cn=10, nknots_ge=5, dtype=tf.double,
	max_lines=150):
	'''
	removes biases due to copy number by aligning copy nunber segments to the mean. Changes the gene_effect matrix.
	Parameters:
		gene_map (`pandas.DataFrame`): either a CCDS gene alignment or a DepMap style guide alignment with the column
										"genome_alignment". Must include all genes.
		copy_number ('pandas.DataFrame'): a cell-line by gene matrix of relative (floating point) copy number
	'''

	if len(gene_effect) < 3:
		raise RuntimeError("Correct for CN should not be used with fewer than 3 cell lines. Consider preprocessing with CRISPRCleanR")
	missing_lines = sorted(set(gene_effect.index) - set(copy_number.index))
	if len(missing_lines) > 0:
		print("Warning: missing lines from gene_effect in copy_number, which won't be corrected.\nExamples: %r" % missing_lines[:5])
	missing_genes = sorted(set(gene_effect.columns) - set(copy_number.columns))
	if len(missing_genes) > 0:
		raise ValueError("Missing %i genes from gene_effect in copy_number.\nExamples: %r" % (
			len(missing_genes), missing_genes[:5]))

	lines = list(gene_effect.index)
	np.random.shuffle(lines)
	ngroups = int(len(gene_effect) / max_lines) + 1
	groups = np.array_split(lines, ngroups)
	shift_list = []
	new_list = []
	for i, group in enumerate(groups):
		print("\nFitting cell line group %i of %i" % (i+1, len(groups)))
	
		print('finding low CN gene effect shifts')
		shifts= get_shifts(gene_effect.loc[group], copy_number.loc[group])

		print('smoothing and interpolating cutting toxicity for all genes')
		means = gene_effect.loc[group].mean().sort_values()
		means_expanded = means.loc[shifts.gene].values
		weights, cn_effect = add_global_shift(shifts.cn, shifts.gene_effect_shift, means_expanded, dtype, nknots_cn, nknots_ge)
		shifts['weights'] = weights
		shifts['cn_effect'] = cn_effect
		shifts['adjusted'] = shifts['gene_effect_shift'].values - cn_effect


		print("generating matrix")
		new = get_adjusted_matrix(shifts, gene_effect.loc[group])
		new_list.append(new)
		shift_list.append(shifts)
	shifts = pd.concat(shift_list, ignore_index=True)
	new = pd.concat(new_list)
	return new, shifts
