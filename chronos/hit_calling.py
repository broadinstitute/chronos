import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from .model import Chronos, check_inputs
from .reports import sum_collapse_dataframes
from warnings import warn
from itertools import permutations


def cell_line_log_likelihood(model):
	'''get the log likelihood of the data from the Chronos model summed to the cell line/gene level and combined
	over libraries. Note the negative sign, since `Chronos.cost_presum` is the negative log likelihood'''
	cost_presum = model.cost_presum
	cost_presum = [
		v\
			.groupby(model.sequence_map[key].set_index("sequence_ID")['cell_line_name']).sum()\
			.groupby(model.guide_gene_map[key].set_index("sgrna")["gene"], axis=1).sum()
		for key, v in cost_presum.items()
	]
	return -sum_collapse_dataframes(cost_presum)


def empirical_pvalue(observed, null, direction=-1):
	'''
	get non-parametric pvalues given a null. Pvalues are calculated individually per column in the null
	Parameters:
		`observed` (1D array-like): observed values for the samples
		`null` (1D array-like): values sampled from the null hypothesis
		 `direction` (-1 or 1), which tail is being tested. -1 tests the hyothesis that observed values are less 
			positive than would be expected by chance
	Returns:
		numpy.ndarray or pandas.Series of pvalues
	'''
	if direction not in [-1, 1]:
		raise ValueError("`direction` must be -1 or 1")

	observed_array = -direction*np.array(observed)
	null_array = -direction*np.array(null)
	sort_observed = np.argsort(observed_array)
	sort_null = np.argsort(null_array)
	observed_array = observed_array[sort_observed]
	null_array = null_array[sort_null]

	pvals = (1+np.searchsorted(null_array, observed_array, side="right"))/(1+len(null_array))
	#undo the sort to get pvals in the same order as observed
	pvals = pvals[np.argsort(sort_observed)]
	if isinstance(observed, pd.Series):
		pvals = pd.Series(pvals, index=observed.index)
	return pvals


def get_difference_significance(observed, null, tail):
	if tail == "both":
		pvals = 2*np.minimum(
				empirical_pvalue(observed, null, direction=-1), 
				empirical_pvalue(observed, null, direction=1)
		)
	elif tail == "right":
		pvals = empirical_pvalue(observed, null, direction=1)
	elif tail == "left":
		pvals = empirical_pvalue(observed, null, direction=-1)
	else:
		raise ValueError("`tail` must be one of 'left', 'right', 'both'")

	fdr = fdrcorrection(pvals, .05)[1]
	return pd.DataFrame({"observed_statistic": observed, "pval": pvals, "FDR": fdr})


def get_difference_between_conditions(nondistinguished, baseline, alt):
	return alt - baseline


def get_difference_distinguished(nondistinguished, baseline, alt):
	return alt + baseline - nondistinguished


def filter_sequence_map_by_condition(condition_map, condition_pair=None):
	out = condition_map.copy()
	if not 'replicate' in out.columns:
		out['replicate'] = out['sequence_ID']
		
	if condition_pair is None:
		condition_pair = out.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")
	if set(condition_pair) - set(out.condition):
		raise ValueError("one or more entries in `condition_pair` %r not present in `condition_map.condition`" % condition_pair)

	condition_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby("cell_line_name")\
					.condition\
					.nunique()
	if (condition_counts < 2).any():
		to_drop = condition_counts.loc[lambda x: x!=2].index
		warn("the following cell lines did not have replicates in both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]
		
	replicate_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby(["cell_line_name", "condition"])\
					.replicate\
					.nunique()
	if (replicate_counts < 2).any():
		to_drop = set(replicate_counts.loc[lambda x: x<2].index.get_level_values(0))
		warn("the following cell lines did not have at least 2 replicates for both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]

	out = out[out.condition.isin(condition_pair) | (out.cell_line_name == 'pDNA')]
	return out

def create_condition_sequence_map(condition_map, condition_pair=None):
	out = filter_sequence_map_by_condition(condition_map, condition_pair)
	out['true_cell_line_name'] = out['cell_line_name'].copy()
	
	def cell_line_overwrite(x):
		if x.cell_line_name == 'pDNA':
			return 'pDNA'
		return '%s__in__%s' % (x.cell_line_name, x.condition)
	out['cell_line_name'] = out.apply(cell_line_overwrite, axis=1)
	
	return out


def unique_permutations(array):
	out = []
	for v in permutations(array):
		arr = np.array(v)
		if any([np.all(arr == o) for o in out]) or np.all(arr == array) or np.none(arr == array):
			continue
		out.append(arr)
	return out


def unique_permutations_no_reversed(array):
	out = []
	for v in permutations(array):
		arr = np.array(v)
		if any([np.all(arr == o) for o in out]) or np.all(arr == array) \
				or np.none(arr == array) or any([np.none(arr == o) for o in out]):
			continue
		out.append(arr)
	return out


	
def create_permuted_sequence_maps(condition_map, condition_pair=None):
	base = filter_sequence_map_by_condition(condition_map, condition_pair)

	if condition_pair is None:
		condition_pair = base.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")
		
	#allow us to detect and exclude permutatations that exactly reverse the label assignments
	base["reverse_condition"] = np.nan
	for i in range(2):
		ind = base[base.condition == condition_pair[i]].index
		base.loc[ind, "reverse_condition"] = condition_pair[1-i]
	
	splits = base.groupby('cell_line_name')
	stack = {}
	for line, y, in splits:
		y['cell_line_name'] = line
		
		if line == 'pDNA':
			stack.append(y)
			continue
			
		condition_replicate_sum = y.groupby("replicate").condition.nunique()
		if (condition_replicate_sum > 1).any():
			raise ValueError("a sequence map with the same replicate of a cell line %s in multiple conditions was passed.\
Replicates must be unique to the condition.\n%r" %(line, y))
			
		original = y['condition'].copy()
		original_reversed = y["reverse_condition"].copy()


		while (y['condition'] == original).all() or (y['condition'] == original_reversed).all():
			np.random.shuffle(y['condition'].values)
		stack.append(y)
		
	return(create_condition_sequence_map(pd.concat(stack, axis=0, ignore_index=True)))


def check_condition_map(condition_map):
	expected_columns = ['sequence_ID', 'cell_line_name', 'pDNA_batch', 'days', 'condition']
	for key in condition_map.keys():
		missing = sorted(set(expected_columns) - set(condition_map[key].columns))
		if missing:
			raise ValueError(
				"`condition_map[%s]` missing expected columns %r" % (key, missing)
			)



class ConditionComparison():
	comparison_effect_dict = {
			"gene_effect": lambda model: model.gene_effect,
			"likelihood": cell_line_log_likelihood
	}
	comparison_statistic_dict = {
			"gene_effect": get_difference_between_conditions,
			"likelihood": get_difference_distinguished
	}

	def __init__(self, readcounts, condition_map, guide_gene_map, **kwargs):
		check_condition_map(condition_map)
		check_inputs(readcounts, guide_gene_map, condition_map)
		self.readcounts = readcounts
		self.condition_map = condition_map
		self.guide_gene_map = guide_gene_map
		self.kwargs = kwargs
		self.keys = sorted(self.readcounts.keys())


	def _check_condition_pair(self,  condition_pair):
		if condition_pair is None:
			condition_pairs = {key: 
				sorted(set(self.condition_map[self.keys[0]].condition.dropna().unique()))
				for key in self.keys
			}
			if not all([
				all([
					condition_pairs[key][i] == condition_pairs[self.keys[0]][i]
					for i in range(2)
				]) and (len(condition_pairs[key]) == 2)
				for key in self.keys
			]):
				raise ValueError("if `condition_pair` is not provided, all the maps in \
`condition_map` must have exactly 2 nonnull unique values, and they must be the same in\
every map.")

			condition_pair = condition_pairs[self.keys[0]]
		if len(condition_pair) != 2:
			raise ValueError("if `condition_pair` is provided, it must have length 2.")

		return condition_pair


	def compare_conditions(self, condition_pair=None, comparison_effect="likelihood",
		comparison_statistic=None, tail="right",
			n_null_iterations=10, n_readcount_total_bins=10, 
				**kwargs):

		if comparison_statistic is None and isinstance(comparison_effect, str):
			comparison_statistic = comparison_effect
		if isinstance(comparison_effect, str):
			comparison_effect = self.comparison_effect_dict[comparison_effect]
		if isinstance(comparison_statistic, str):
			comparison_statistic = self.comparison_statistic_dict[comparison_statistic]
		if not callable(comparison_effect):
			raise ValueError("`comparison_effect` must be a callable that accepts `Chronos`\
isntances or one of %r" % list(self.comparison_effect_dict.keys())
			)
		if not callable(comparison_statistic):
			raise ValueError("`comparison_statistic` must be a callable that accepts 3 \
`pandas.Series` or %r" % list(self.comparison_statistic_dict.keys())
			)

		condition_pair = self._check_condition_pair(condition_pair)

		print("training model with no condition distincions\n")
		self.nondistinguished_map, self.retained_readcounts,\
			 self.nondistinguished_result, nondistinguished_gene_effect = self.get_nondistinguished_results(
			condition_pair, comparison_effect, **kwargs
		)
		self.compared_lines = self.nondistinguished_result.index

		print("training model with conditions distinguished")
		self.distinguished_map, self.distinguished_result, \
			distinguished_gene_effect = self.get_distinguished_results(
			self.nondistinguished_map, condition_pair, comparison_effect, **kwargs
		)

		print("training model with permuted conditions")
		self.permuted_maps, self.permuted_results, \
			permuted_gene_effects = self.get_permuted_results(
			n_null_iterations, self.nondistinguished_map, condition_pair, 
			comparison_effect, **kwargs
		)

		print("calculating statistics")
		self.observed_statistic, self.permuted_statistics = self.get_comparison_statistic(
			self.nondistinguished_result, self.distinguished_result, self.permuted_results,
			comparison_statistic, condition_pair
		)


		gene_effect_in_alt = distinguished_gene_effect.loc[[
			'%s__in__%s' % (line, condition_pair[1])
			for line in self.compared_lines
		]]
		gene_effect_in_alt.set_index(np.array(self.compared_lines), inplace=True)

		gene_effect_in_baseline = distinguished_gene_effect.loc[[
			'%s__in__%s' % (line, condition_pair[0])
			for line in self.compared_lines
		]]
		gene_effect_in_baseline.set_index(np.array(self.compared_lines), inplace=True)

		gene_effect_difference = gene_effect_in_alt - gene_effect_in_baseline

		gene_effect_in_alt_permuted = [
			permuted_gene_effect.loc[[
				'%s__in__%s' % (line, condition_pair[1])
				for line in self.compared_lines
			]].set_index(np.array(self.compared_lines))
		for permuted_gene_effect in permuted_gene_effects]
		
		gene_effect_in_baseline_permuted = [
			permuted_gene_effect.loc[[
				'%s__in__%s' % (line, condition_pair[0])
				for line in self.compared_lines
			]].set_index(np.array(self.compared_lines))
		for permuted_gene_effect in permuted_gene_effects]

		gene_effect_difference_permuted = [
			gene_effect_in_alt_permuted[i] - gene_effect_in_baseline_permuted[i]
			for i in range(len(gene_effect_in_baseline_permuted))
		]

		gene_effect_annotations = {
			"gene_effect_nondistinguished": nondistinguished_gene_effect.stack(),
			"gene_effect_in_%s" % condition_pair[0]: gene_effect_in_baseline.stack(),
			"gene_effect_in_%s" % condition_pair[1]: gene_effect_in_alt.stack(),
			"gene_effect_difference": gene_effect_difference.stack(),

			"permuted_gene_effect_in_%s_min" % condition_pair[0]: pd.DataFrame(
				np.min(
					gene_effect_in_baseline_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			).stack(),

			 "permuted_gene_effect_in_%s_min" % condition_pair[1]: pd.DataFrame(
			 	np.min(
					gene_effect_in_alt_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			 ).stack(),

			"permuted_gene_effect_in_%s_max" % condition_pair[0]: pd.DataFrame(
				np.max(
					gene_effect_in_baseline_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			).stack(),

			 "permuted_gene_effect_in_%s_max" % condition_pair[1]: pd.DataFrame(
			 	np.max(
					gene_effect_in_alt_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			 ).stack(),

			"permuted_gene_effect_in_%s_mean" % condition_pair[0]: pd.DataFrame(
				np.mean(
					gene_effect_in_baseline_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			).stack(),

			 "permuted_gene_effect_in_%s_mean" % condition_pair[1]: pd.DataFrame(
			 	np.mean(
					gene_effect_in_alt_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			 ).stack(),

			"permuted_difference_sd": pd.DataFrame(
			 	np.std(
					gene_effect_difference_permuted,
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			 ).stack(),

			"permuted_difference_extreme": pd.DataFrame(
			 	np.max(
					np.abs(gene_effect_difference_permuted),
					axis=0
				), index=self.compared_lines, columns=gene_effect_in_baseline_permuted[0].columns
			 ).stack()

		}
		for key, v in gene_effect_annotations.items():
			print(key)
			print(v[:3])
			print(v.shape)
			print()
		gene_effect_annotations = pd.DataFrame(gene_effect_annotations).reset_index()
		gene_effect_annotations.rename(columns={
			gene_effect_annotations.columns[0]: "cell_line_name",
			gene_effect_annotations.columns[1]: "gene",
		}, inplace=True)


		print("calculating empirical significance")
		statistics = self.get_significance_by_readcount_bin(
			self.retained_readcounts, self.distinguished_map,
			self.compared_lines,
			self.observed_statistic, self.permuted_statistics,
			tail, n_readcount_total_bins
		)

		return gene_effect_annotations.merge(statistics, on=["cell_line_name", "gene"],
			how="outer")



	def get_nondistinguished_results(self, condition_pair,
			comparison_effect, **kwargs):
		nondistinguished_map = {key: filter_sequence_map_by_condition(
			self.condition_map[key], condition_pair)
			for key in self.keys
		}
		retained_readcounts = {
			key:self.readcounts[key].loc[nondistinguished_map[key].sequence_ID]
			for key in self.keys
		}
		nondistinguished_model = Chronos(
			 readcounts=retained_readcounts,
			 sequence_map=nondistinguished_map,
			 guide_gene_map=self.guide_gene_map,
			 **self.kwargs
		)
		nondistinguished_model.train(**kwargs)

		nondistinguished_result = comparison_effect(nondistinguished_model)
		nondistinguished_gene_effect = nondistinguished_model.gene_effect
		del nondistinguished_model

		return nondistinguished_map, retained_readcounts, nondistinguished_result,\
			nondistinguished_gene_effect


	def get_distinguished_results(self, nondistinguished_map, condition_pair,
			comparison_effect, **kwargs
		):

		distinguished_map = {key:
			create_condition_sequence_map(nondistinguished_map[key], condition_pair)
			for key in self.keys}
		distinguished_model = Chronos(
			readcounts=self.retained_readcounts,
			sequence_map=distinguished_map,
			guide_gene_map=self.guide_gene_map,
			**self.kwargs
		)
		distinguished_model.train(**kwargs)

		distinguished_result = comparison_effect(distinguished_model)
		distinguished_gene_effect = distinguished_model.gene_effect
		del distinguished_model

		return distinguished_map, distinguished_result, distinguished_gene_effect


	def get_permuted_results(self, n_iterations, nondistinguished_map, condition_pair, 
			comparison_effect, **kwargs
		):

		out = []
		permuted_maps = []
		permuted_gene_effects = []
		for i in range(n_iterations):
			print('\n\n&&&&&&&&&&&&&&&&\nrandom iteration %i\n&&&&&&&&&&&&&&&&\n\n' %i)

			permuted_map = {
				key: create_randomized_sequence_map(
					nondistinguished_map[key], 
					condition_pair
				)
				for key in self.keys
			}

			if any([
				any([
					(permuted_map[key].condition.dropna() == m[key].condition.dropna()).all()
					for key in self.keys
				]) 
				for m in permuted_maps
			]):
				continue
				
			permuted_model = Chronos(readcounts=self.retained_readcounts, 
								 sequence_map=permuted_map,
								 guide_gene_map=self.guide_gene_map,
								**self.kwargs
								)
			permuted_model.train(**kwargs)

			out.append(comparison_effect(permuted_model))
			permuted_maps.append(permuted_map)
			permuted_gene_effects.append(permuted_model.gene_effect)
			del permuted_model

		return permuted_maps, out, permuted_gene_effects


	def get_comparison_statistic(self, nondistinguished_result, distinguished_result, 
			permuted_results, statistic, condition_pair
		):

		observed_statistic = {}
		permuted_statistics = []
		for line in self.compared_lines:
			baseline = '%s__in__%s' % (line, condition_pair[0])
			alt = '%s__in__%s' % (line, condition_pair[1])

			observed_statistic[line] = statistic(
				nondistinguished_result.loc[line],
				distinguished_result.loc[baseline],
				distinguished_result.loc[alt]
			)

			permuted_diff = pd.DataFrame([
				statistic(
					nondistinguished_result.loc[line],
					permuted_result.loc[baseline],
					permuted_result.loc[alt]
				)
				for permuted_result in permuted_results
			])
			permuted_diff.index = [line] * len(permuted_diff)
			permuted_statistics.append(permuted_diff)

		permuted_statistics = pd.concat(permuted_statistics)
		observed_statistic = pd.DataFrame(observed_statistic).T

		return observed_statistic, permuted_statistics



	def get_significance_by_readcount_bin(self, retained_readcounts,
		distinguished_map, compared_lines, observed_statistic, permuted_statistics, 
			tail, nbins=10,  additional_annotations={}
		):

		readcount_gene_totals = sum_collapse_dataframes([
			retained_readcounts[key]\
						.groupby(self.guide_gene_map[key].set_index("sgrna")["gene"], axis=1)\
						.sum()\
						.sum(axis=0)
			for key in self.keys
		])

		bins = readcount_gene_totals.quantile(np.linspace(0, 1.0, nbins+1))
		bins[0] = 0
		bins[1.0] *= 1.05
		bins = pd.cut(readcount_gene_totals, bins)
		out = []

		for line in compared_lines:
			if line == 'pDNA':
				continue
			for bin in sorted(bins.unique()):
				genes = bins.loc[lambda x: x==bin].index

				null = permuted_statistics.loc[line, genes]
				if len(null.shape) > 1:
					null_mean = null.mean(axis=0)
					null_max = null.max(axis=0)
					null_min = null.min(axis=0)
					null = null.stack()
				else:
					null_mean, null_min, null_max = null

				try:
					out.append(get_difference_significance(
						observed_statistic.loc[line, genes], 
						null,
						tail
					))
				except:
					print(observed_statistic.loc[line, genes])
					print(permuted_statistics.loc[line, genes])
					assert False
				out[-1]["cell_line_name"] = line
				out[-1]["permuted_mean_statistic"] = null_mean
				out[-1]["permuted_min_statistic"] = null_min
				out[-1]["permuted_max_statistic"] = null_max

				for key, val in additional_annotations:
					if isinstance(val, pd.Series):
						try:
							val = val.loc[line].loc[genes]
						except IndexError:
							raise ValueError("additional annotation '%s' missing genes:\n%r" %
								(key, val))
					out[-1][key] = val
				out[-1].reset_index(inplace=True)
				out[-1].rename(columns={out[-1].columns[0]: "gene"})
		return pd.concat(out, ignore_index=True)

