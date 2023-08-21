import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from .model import Chronos, check_inputs
from .reports import sum_collapse_dataframes



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


def get_difference_statistics(observed, null):
	pvals = 2*np.minimum(
			empirical_pvalue(observed, null, direction=-1), 
			empirical_pvalue(observed, null, direction=1)
	)
	fdr = fdrcorrection(pvals, .05)[1]
	return pd.DataFrame({"Observed": observed, "pval": pvals, "FDR": fdr})


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

	
def create_randomized_sequence_map(condition_map, condition_pair=None):
	out = filter_sequence_map_by_condition(condition_map, condition_pair)

	if condition_pair is None:
		condition_pair = out.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")
		
	#allow us to detect and exclude permutatations that exactly reverse the label assignments
	out["reverse_condition"] = np.nan
	for i in range(2):
		ind = out[out.condition == condition_pair[i]].index
		out.loc[ind, "reverse_condition"] = condition_pair[1-i]
	
	splits = out.groupby('cell_line_name')
	stack = []
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

	def __init__(self, readcounts, condition_map, guide_gene_map, **kwargs):
		check_condition_map(condition_map)
		check_inputs(readcounts, guide_gene_map, condition_map)
		self.readcounts = readcounts
		self.condition_map = condition_map
		self.guide_gene_map = guide_gene_map
		self.kwargs = kwargs
		self.keys = sorted(self.readcounts.keys())

		self.comparison_function_dict = {
			"gene_effect": lambda model: model.gene_effect
		}

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


	def compare_conditions(self, condition_pair=None, comparison_value="gene_effect",
			n_null_iterations=10, n_readcount_total_bins=10,
				**kwargs):
		if isinstance(comparison_value, str):
			comparison_value = self.comparison_function_dict[comparison_value]
		if not callable(comparison_value):
			raise ValueError("`comparison_value must be a callable or one of %r"
				% list(self.comparison_function_dict.keys())
			)

		condition_pair = self._check_condition_pair(condition_pair)

		print("training model with no condition distincions\n")
		self.nondistinguished_map, self.retained_readcounts,\
			 self.nondistinguished_result = self.get_nondistinguished_results(
			condition_pair, comparison_value, **kwargs
		)
		self.compared_lines = self.nondistinguished_result.index

		print("training model with conditions distinguished")
		self.distinguished_map, self.distinguished_result= self.get_distinguished_results(
			self.nondistinguished_map, condition_pair, comparison_value, **kwargs
		)

		print("training model with shuffled conditions")
		self.shuffled_maps, self.shuffled_results = self.get_shuffled_results(
			n_null_iterations, self.nondistinguished_map, condition_pair, 
			comparison_value, **kwargs
		)

		print("calculating statistics")
		observed_comparison = {}
		shuffled_comparisons = []
		for line in self.compared_lines:
			baseline = '%s__in__%s' % (line, condition_pair[0])
			alt = '%s__in__%s' % (line, condition_pair[1])
			observed_comparison[line] = self.distinguished_result.loc[alt] \
										- self.distinguished_result.loc[baseline]
			shuffled_diff = pd.DataFrame([
				shuffled_result.loc[alt] - shuffled_result.loc[baseline]
				for shuffled_result in self.shuffled_results
			])
			shuffled_diff.index = [line] * len(shuffled_diff)
			shuffled_comparisons.append(shuffled_diff)

		self.shuffled_comparisons = pd.concat(shuffled_comparisons)
		self.observed_comparison = pd.DataFrame(observed_comparison).T

		statistics = self.get_statistics_by_readcount_bin(
			self.retained_readcounts, self.distinguished_map,
			self.compared_lines,
			self.observed_comparison, self.shuffled_comparisons,
			n_readcount_total_bins
		)

		return statistics



	def get_nondistinguished_results(self, condition_pair,
			comparison_value, **kwargs):
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
		nondistinguished_result = comparison_value(nondistinguished_model)
		del nondistinguished_model
		return nondistinguished_map, retained_readcounts, nondistinguished_result


	def get_distinguished_results(self, nondistinguished_map, condition_pair,
			comparison_value, **kwargs
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
		distinguished_result = comparison_value(distinguished_model)
		del distinguished_model
		return distinguished_map, distinguished_result


	def get_shuffled_results(self, n_iterations, nondistinguished_map, condition_pair, 
			comparison_value, **kwargs
		):

		out = []
		shuffled_maps = []
		for i in range(n_iterations):
			print('\n\n&&&&&&&&&&&&&&&&\nrandom iteration %i\n&&&&&&&&&&&&&&&&\n\n' %i)

			shuffled_map = {
				key: create_randomized_sequence_map(
					nondistinguished_map[key], 
					condition_pair
				)
				for key in self.keys
			}

			if any([
				any([
					(shuffled_map[key].condition.dropna() == m[key].condition.dropna()).all()
					for key in self.keys
				]) 
				for m in shuffled_maps
			]):
				continue
				
			shuffled_model = Chronos(readcounts=self.retained_readcounts, 
								 sequence_map=shuffled_map,
								 guide_gene_map=self.guide_gene_map,
								**self.kwargs
								)
			shuffled_model.train(**kwargs)

			out.append(comparison_value(shuffled_model))
			shuffled_maps.append(shuffled_map)
			del shuffled_model

		return shuffled_maps, out


	def get_statistics_by_readcount_bin(self, retained_readcounts,
		distinguished_map, compared_lines, observed_comparison, shuffled_comparisons, 
			nbins=10, additional_annotations={}
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
				null = shuffled_comparisons.loc[line, genes]
				if len(null.shape) > 1:
					null = null.stack()
				try:
					out.append(get_difference_statistics(
						observed_comparison.loc[line, genes], 
						null
					))
				except:
					print(observed_comparison.loc[line, genes])
					print(shuffled_comparisons.loc[line, genes])
					assert False
				out[-1]["cell_line_name"] = line
				for key, val in additional_annotations:
					if isinstance(val, pd.Series):
						try:
							val = val.loc[genes]
						except IndexError:
							raise ValueError("additional annotation '%s' missing genes:\n%r" %
								(key, val))
					out[-1][key] = val
				out[-1].reset_index(inplace=True)
				out[-1].rename(columns={out[-1].columns[0]: "gene"})
		return pd.concat(out, ignore_index=True)

