import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from .model import Chronos, check_inputs
from .reports import sum_collapse_dataframes
from warnings import warn
from itertools import permutations
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from sympy.utilities.iterables import multiset_permutations


def cell_line_log_likelihood(model, distinguished_condition_map):
	'''get the log likelihood of the data from the Chronos model summed to the cell line/gene level and combined
	over libraries. Note the negative sign, since `Chronos.cost_presum` is the negative log likelihood'''
	cost_presum = model.cost_presum
	cost_presum = [
		v\
			.groupby(distinguished_condition_map[key].set_index("sequence_ID")['true_cell_line_name']).sum()\
			.groupby(model.guide_gene_map[key].set_index("sgrna")["gene"], axis=1).sum()

		for key, v in cost_presum.items()
	]
	return -sum_collapse_dataframes(cost_presum)


def unique_permutations(array):
	out = list(np.array([v for v in multiset_permutations(array)]))
	#exclude the trivial permutation
	return [v for v in out if not np.all(v == array)]


def unique_permutations_no_reversed(array):
	unique = np.unique(array)
	assert len(unique) == 2, "Array must have exactly two unique values"
	mapper = {val: bool(i) for i, val in enumerate(unique)}
	bool_array = np.array([mapper[val] for val in array])
	perms = unique_permutations(bool_array)
	out = []
	mirror = set([
		tuple(bool_array),
		tuple(~bool_array)
		])
	for perm in perms:
		reverse = tuple(~perm)
		if not reverse in mirror:
			mirror.add(tuple(perm))
			out.append(unique[perm.astype(int)])
	return out



def reverse_gene_effect(gene_effect, distinguished_condition_map, condition_pair):
	'''reverse the condition labels for the gene effect matrix index'''
	out = gene_effect.copy()

	if isinstance(distinguished_condition_map, dict):
		lines = sorted(set.union(*[set(v.true_cell_line_name) 
			for v in distinguished_condition_map.values()]))
	else:
		lines = list(distinguished_condition_map.true_cell_line_name.unique())
	lines.remove("pDNA")

	for line in lines:

		out.loc[
			'%s__in__%s' % (line, condition_pair[0])
		] = gene_effect.loc['%s__in__%s' % (line, condition_pair[1])]

		out.loc[
			'%s__in__%s' % (line, condition_pair[1])
		] = gene_effect.loc['%s__in__%s' % (line, condition_pair[0])]

	return out


def change_in_likelihood(model, distinguished_condition_map, condition_pair):
	'''
	Get the change in likelihood for a `model` when the gene effect of each cell line
	in each condition is reversed
	'''
	likelihood_baseline = cell_line_log_likelihood(model, distinguished_condition_map)
	ge = model.gene_effect
	model.gene_effect = reverse_gene_effect(ge, distinguished_condition_map, condition_pair)
	likelihood_reversed = cell_line_log_likelihood(model, distinguished_condition_map)
	model.gene_effect = ge
	return likelihood_baseline - likelihood_reversed


def change_in_gene_effect(model, distinguished_condition_map, condition_pair):
	'''get the differences in gene effect for the same lines in the first vs second condition'''
	ge = model.gene_effect
	# get unique lines allowing for someone deciding to put "__in__" in their base cell line name
	all_lines = set(['__in__'.join(s.split('__in__')[:-1]) for s in ge.index])
	out = {
		line: ge.loc["%s__in__%s" % (line, condition_pair[1])]
			- ge.loc["%s__in__%s" % (line, condition_pair[0])]
		for line in all_lines
	}
	return pd.DataFrame(out).T



def empirical_pvalue(observed, null, direction=-1):
	'''
	Returns an array or `pandas.Series` of non-parametric pvalues given a null. 
	For a given observation in `observed`, `p = (n_null_more_extreme + 1) / (n_null + 1)`,
	where n_null_more_extreme is the number of null observations equal or more extreme
	that the observation and n_null is `len(null)`.
	Parameters:
		`observed` (1D array-like): observed values for the samples
		`null` (1D array-like): values sampled from the null hypothesis
		`direction` (-1 or 1), which tail is being tested. -1 tests the hyothesis that 
		the observed values are less positive than would be expected by chance
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


################################################################
# C O M P A R E    C O N D I T I O N S
################################################################


def get_difference_significance(observed, null, tail):
	'''
	Combines effect size, p-value, and FDR in one dataframe
	'''
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


def filter_sequence_map_by_condition(condition_map, condition_pair=None):
	'''
	Given a pair of conditions, removes cell lines from the `condition_map` without
	at least two replicates per condition and returns the new map. Replicates are filtered
	so that there is an even and equal number in each condition.
	'''
	out = condition_map.copy()
		
	if condition_pair is None:
		condition_pair = out.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")
	if set(condition_pair) - set(out.condition):
		raise ValueError("one or more entries in `condition_pair` %r not present in `condition_map.condition`" 
			% list(condition_pair))

	condition_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby("cell_line_name")\
					.condition\
					.nunique()
	if (condition_counts < 2).all():
		raise ValueError("No cell lines in `condition_map` had replicates in both conditions \
in `condition_pair`: %r/n/n%r" % (condition_pair, condition_map))
	if (condition_counts < 2).any():
		to_drop = condition_counts.loc[lambda x: x!=2].index
		warn("the following cell lines did not have replicates in both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]
		
	replicate_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby(["cell_line_name", "condition"])\
					.sequence_ID\
					.nunique()
	if (replicate_counts < 2).all():
		raise ValueError("No cell lines in `condition_map` had at least 2 replicates in both conditions \
in `condition_pair`: %r/n/n%r" % (condition_pair, condition_map))
	if (replicate_counts < 2).any():
		to_drop = set(replicate_counts.loc[lambda x: x<2].index.get_level_values(0))
		warn("the following cell lines did not have at least 2 replicates for both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]

	out = out[out.condition.isin(condition_pair) | (out.cell_line_name == 'pDNA')]

	#balance the number of replicates in each condition
	retain_sequences = []
	for line, group in out.groupby("cell_line_name"):
		if line == 'pDNA':
			retain_sequences.extend(list(group.sequence_ID))
			continue
		n_replicates = group.groupby("condition").replicate.nunique().min()
		if n_replicates % 2:
			n_replicates -= 1
		for condition, subgroup in group.groupby("condition"):
			retain_replicates = subgroup.replicate.unique()[:n_replicates]
			retain_sequences.extend(list(
				subgroup\
					.query("replicate in %r" % list(retain_replicates))\
					.sequence_ID
			))

	return out[out.sequence_ID.isin(retain_sequences)].copy()


def create_condition_sequence_map(condition_map, condition_pair=None):
	'''
	Returns a new map filtered for `condition` in `condition_pair` with cell lines having fewer
	than 2 replicates in either condition removed, where `cell_line_name` has "__in__<condition>" appended.
	'''
	out = condition_map.copy()
	out['true_cell_line_name'] = out['cell_line_name'].copy()
	
	def cell_line_overwrite(x):
		if x.cell_line_name == 'pDNA':
			return 'pDNA'
		return '%s__in__%s' % (x.cell_line_name, x.condition)
	out['cell_line_name'] = out.apply(cell_line_overwrite, axis=1)
	
	return out

	
def create_permuted_sequence_maps(condition_map, condition_pair=None, allow_reverse=False):
	'''
	Returns a list of condition maps, where each cell line within each map has a unique permutation
	of condition labels which have been appended to `cell_line_name`. The original condition and its 
	mirror image are never returned. If `allow_reverse` is False,
	mirroring permutations are discarded. E.g. if `condition` in `condition_map` is ['A', 'A', 'B', 'B'],
	only ['A', 'B', 'A', 'B'] and ['A', 'B', 'B', 'A'] are returned.
	Additionally, only permutations with an equal number of replicates from each condition are retained.
	'''
	seq_map = filter_sequence_map_by_condition(condition_map, condition_pair)

	if condition_pair is None:
		condition_pair = seq_map.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")

	#drop days column for identifying possible permutations
	days_map = seq_map[['cell_line_name','days']].drop_duplicates()
	base = seq_map.drop(columns=['days','sequence_ID']).drop_duplicates()
	
	splits = base.groupby('cell_line_name')
	stack = {}
	for line, y, in splits:
		y['cell_line_name'] = line
		
		if line == 'pDNA':
			pdna = y.copy()
			pdna["true_condition"] = pdna["condition"]
			continue

		stack[line] = []
		if allow_reverse:
			perms = unique_permutations(y.condition)
		else:
			perms = unique_permutations_no_reversed(y.condition)

		for perm in perms:
				tentative = y.copy()
				tentative["true_condition"] = tentative["condition"]
				tentative["condition"] = perm
				tentative["cell_line_name"] = line

				#check that the pseudo conditions have equal numbers of replicates from each real condition
				condition_counts = tentative.groupby("condition").true_condition.value_counts()
				if \
						len(condition_counts) == 4 \
					 and condition_counts.min() == condition_counts.max():
					stack[line].append(tentative)

	min_unique_permutations = min(len(v) for v in stack.values())
	out = []
	for i in range(min_unique_permutations):
		#inner merge is to incl. row for each day (in case replicate has data for multiple days)
		#outer merge is to add back sequence ids for each row
		out.append(
			pd.merge(
				pd.merge(
					pd.concat([v[i] for v in stack.values()] + [pdna]), 
					days_map
				),
				seq_map.rename(columns={'condition':'true_condition'}) 
			)
		) 

	return [create_condition_sequence_map(v, condition_pair) for v in out]


def check_condition_map(condition_map):
	expected_columns = ['sequence_ID', 'cell_line_name', 'pDNA_batch', 'days', 'replicate', 'condition']
	for key in condition_map.keys():
		missing = sorted(set(expected_columns) - set(condition_map[key].columns))
		if missing:
			raise ValueError(
				"`condition_map[%s]` missing expected columns %r" % (key, missing)
			)




class ConditionComparison():
	'''
	An object that manages the various Chronos models needed to compare two conditions. 
	The strategy is to compare the chosen measure of difference between the two conditions
	per gene to a null distribution generated by permuting which replicates in each cell line
	are assigned to each condition, generating an empirical p-value. The key method that returns
	the statistics is `compare_conditions`.
	'''
	comparison_effect_dict = {
			"gene_effect": change_in_gene_effect,
			"likelihood": change_in_likelihood
	}
	# which tails to test the p-value in for each comparison effect
	comparison_effect_tail_dict = {
			"gene_effect": "both",
			"likelihood": "right"
	}

	def __init__(self, readcounts, condition_map, guide_gene_map,
		print_to=None, **kwargs):
		'''
		Initialize the comparator.
		Parameters:
			`readcounts` (`dict` of `pandas.DataFrame`): readcount matrices from the experiment.
					See `model.Chronos`
			`condition_map` (`dict` of `pandas.DataFrame`): Tables in the same format as `sequence_map`
					for `model.Chronos`, but with the additional columns `replicate` (e.g. A, B), and 
					`condition`, which the comparator will compare results between. `condition` can be 
					any value that can be passed to `str`.
					Results will be reported separately per cell line.
					If you wish to compare two cell lines, give them the same value in `cell_line_name`,
					and different values for `condition`.
			`guide_gene_map` (`dict` of `pandas.DataFrame`): map from sgRNAs to genes. See `model.Chronos`.
			Additional keyword arguments will be passed to `model.Chronos` when training the models.
		'''
		check_condition_map(condition_map)
		check_inputs(readcounts, guide_gene_map, condition_map)
		self.readcounts = readcounts
		self.condition_map = condition_map
		self.guide_gene_map = guide_gene_map
		self.print_to = print_to
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


	def compare_conditions(self, condition_pair=None,
		gene_readcount_total_bin_quantiles=[0.05], 
		allow_reversed_permutations=False,
			max_null_iterations=20,
				**kwargs):
		'''
		Generate a table with the significance of differences in gene effect between two conditions.
		First, a model is trained with no distinction between the different conditions.
		Then, a model is trained distinguishing the different conditions by modifying `cell_line_name`
		in the `sequence_map`s. 
		Finally, the condition labels are permuted within each cell line and a new model is trained
		per permutation. Each cell line in each permutation is required to have a unique permutation
		with respect to the same line in the other permutations, so the total number of permutations
		is limited by the cell line with the fewest replicates. 
		P-values are estimated per gene and cell line  using the distribution of differences between 
		conditions in the models. Permutation results from different genes are grouped together according
		to total reads informing the estimate to increase statistical power. 
		Parameters:
			`condition_pair` (iteranble of len 2 or None): the two conditions to be compared. If None,
				requires that `self.condition_map` have only two uniue conditions for non-pDNA entries.
			`gene_readcount_total_bin_quantiles`: gene effect estimates for genes informed by few total reads are
				noisier than those with abundant reads. Therefore, when calculating p-values, genes
				will be binned by total readcounts according to the quantiles passed here. 0 and 1 are assumed.
				 More bins improves control of false discovery in common essential genes, but at the price of raising the minimum 
				achievable p value (which is 1/number of samples in the null distribution).
			`allow_reversed-permutations` (`bool`): whether to allow permutations that are 
				mirror images of each other - e.g. if ['A', 'B', 'A', 'B'] is one permutation, whether 
				to also include ['B', 'A', 'B', 'A']. Mirror image permutations will cause Chronos to
				estimate highly similar gene effects, just with the condition labels swapped. Thus,
				the permutations are no longer independently distributed. 
			`max_null_permutations` (`int`): limits the number of permutations used in the null, useful
				in the case that there are many replicates in each cell line.
		Returns:
			`statistics` (`pd.DataFrame`): A dataframe with the columns:
				`cell_line_name`: the cell line name from `condition_map`
				`gene`: the gene from `guide_gene_map`
				`gene_effect_nondistinguished`: the Chronos-estimated gene effect found by treating sequences 
					of the same cell line in every condition as a replicate, i.e. not distinguishing on 
					condition.
				`gene_effect_in_<condition1>`, `gene_effect_in_<condition2>`: Chronos estimates of the gene 
					effect for the line treating each condition as a different cell line.
				`gene_effect_difference`: `gene_effect_in_<condition2>` - `gene_effect_in_<condition1>`
				`permuted_gene_effect_in_<condition>_<min, max or mean>`: the results from the permutation
					tests, with the min, max or mean taken over the different permutations.
				`permuted_difference_sd`: The standard deviation of the difference seen between conditions over 
					the different permutations.
				`permuted_difference_extreme`: the maximal absolute difference seen between conditions over the
					the different permutations.
				`observed_statistic`: the value calculated by applying `comparison_statistic` to the
					`comparison_effect`s in the nondistinguished, condition1, and condition2 cases.
				`pval`: the empirical p-value that the observed statistic did not arise from the distribution of
					statistics seen with the permuted condition labels.
				`FDR`: Benjamini-Hochberg estimate from the `pval` distribution (within readcount bins)
				`permuted_<min, max, mean>_statistic`: as `observed_statistic`, summarized over the permutations

		'''

		condition_pair = self._check_condition_pair(condition_pair)

		self.nondistinguished_map = {key: filter_sequence_map_by_condition(
			self.condition_map[key], condition_pair)
			for key in self.keys
		}
		self.retained_readcounts = {
			key:self.readcounts[key].loc[self.nondistinguished_map[key].sequence_ID]
			for key in self.keys
		}
		self.compared_lines = sorted(set.union(*[set(v.cell_line_name) for v in self.nondistinguished_map.values()]))
		self.compared_lines.remove("pDNA")

		self.readcount_gene_totals = self.get_readcount_gene_totals(
			self.retained_readcounts, self.condition_map, self.guide_gene_map
		)

		print("training model with conditions distinguished")
		self.distinguished_map, self.distinguished_result, \
			distinguished_gene_effect = self.get_distinguished_results(
			self.nondistinguished_map, condition_pair,
			self.readcount_gene_totals, **kwargs
		)

		print("training models with permuted conditions")
		self.permuted_maps, self.permuted_results, \
			permuted_gene_effects = self.get_permuted_results(
			max_null_iterations, self.nondistinguished_map, condition_pair, 
			allow_reversed_permutations, **kwargs
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


		gene_effect_annotations = {
			"gene_effect_in_%s" % condition_pair[0]: gene_effect_in_baseline.stack(),
			"gene_effect_in_%s" % condition_pair[1]: gene_effect_in_alt.stack(),
			"gene_effect_difference": gene_effect_difference.stack(),

			
		}
		gene_effect_annotations = pd.DataFrame(gene_effect_annotations).reset_index()
		gene_effect_annotations.rename(columns={
			gene_effect_annotations.columns[0]: "cell_line_name",
			gene_effect_annotations.columns[1]: "gene",
		}, inplace=True)


		print("calculating empirical significance")
		significance = self.get_significance_by_readcount_bin(
			self.readcount_gene_totals, self.distinguished_map,
			self.compared_lines,
			self.distinguished_result, self.permuted_results,
			gene_readcount_total_bin_quantiles
		)

		return gene_effect_annotations\
			.merge(significance, on=["cell_line_name", "gene"], how="outer")


	def get_readcount_gene_totals(self, retained_readcounts, condition_map, guide_gene_map):
		pdna_seqs = {key: condition_map[key].query("cell_line_name == 'pDNA'").sequence_ID
					for key in condition_map}
		readcount_gene_totals = sum_collapse_dataframes([
			retained_readcounts[key]\
						.drop(pdna_seqs[key], axis=0, errors="ignore")\
						.groupby(guide_gene_map[key].set_index("sgrna")["gene"], axis=1)\
						.sum()\
						.median(axis=0)
			for key in self.keys
		])
		return readcount_gene_totals


	def get_distinguished_results(self, nondistinguished_map, condition_pair,
			readcount_gene_totals, **kwargs
		):

		distinguished_map = {key:
			create_condition_sequence_map(nondistinguished_map[key], condition_pair)
			for key in self.keys}


		distinguished_model = Chronos(
			readcounts=self.retained_readcounts,
			sequence_map=distinguished_map,
			guide_gene_map=self.guide_gene_map,
			 use_line_mean_as_reference=np.inf,
			 print_to=self.print_to,
			**self.kwargs
		)
		distinguished_model.train(**kwargs)

		distinguished_gene_effect = distinguished_model.gene_effect
		distinguished_result = {comparison_effect: func(
				distinguished_model, distinguished_map, condition_pair
			)
			for comparison_effect, func in self.comparison_effect_dict.items()
		}
		#del distinguished_model
		self.distinguished_model = distinguished_model

		return distinguished_map, distinguished_result, distinguished_gene_effect


	def get_permuted_results(self, max_null_iterations, nondistinguished_map, condition_pair, 
			allow_reversed_permutations, **kwargs
		):

		permuted_maps = {
			key: create_permuted_sequence_maps(nondistinguished_map[key], condition_pair, 
												allow_reversed_permutations)
			for key in self.keys
		}
		min_permutations = min([len(permuted_maps[key]) for key in self.keys])
		permuted_maps = [{key: permuted_maps[key][i] for key in self.keys} for i in range(min_permutations)]
		out = []
		permuted_gene_effects = []
		counts = 0
		for i, permuted_map in enumerate(permuted_maps):
			print('\n\n&&&&&&&&&&&&&&&&\nrandom iteration %i\n&&&&&&&&&&&&&&&&\n\n' %i)
				
			permuted_model = Chronos(readcounts=self.retained_readcounts, 
								 sequence_map=permuted_map,
								 guide_gene_map=self.guide_gene_map,
								  use_line_mean_as_reference=np.inf,
								  print_to=self.print_to,
								**self.kwargs
								)
			permuted_model.train(**kwargs)

			out.append({comparison_effect: func(
					permuted_model, permuted_map, condition_pair
				)
				for comparison_effect, func in self.comparison_effect_dict.items()
			})
			permuted_gene_effects.append(permuted_model.gene_effect)
			del permuted_model

			counts += 1
			if counts == max_null_iterations:
				break

		return permuted_maps, out, permuted_gene_effects


	def get_significance_by_readcount_bin(self, readcount_gene_totals,
		distinguished_map, compared_lines, observed_statistic, permuted_statistics, 
			gene_readcount_total_bin_quantiles,  additional_annotations={}
		):

		bins = readcount_gene_totals.quantile([0] + gene_readcount_total_bin_quantiles + [1])
		bins[0] = 0
		bins[1.0] *= 1.05
		bins = pd.cut(readcount_gene_totals, bins)
		out = []
		bin_assignments = []

		for line in compared_lines:
			if line == 'pDNA':
				continue
			for bin in sorted(bins.unique()):
				genes = bins.loc[lambda x: x==bin].index
				

				if len(genes) < 100:
					warn("Only %i genes in one of your bins. This will limit the minimum achievable p-value \
. If you have a sub-genome library, considering changing `gene_readcount_total_bin_quantiles` so there are \
more genes in each bin." % (len(genes)))

				significance = []
				for comparison_effect, tail in self.comparison_effect_tail_dict.items():
					null = pd.concat([v[comparison_effect].loc[line, genes] for v in permuted_statistics]) 
					significance.append(get_difference_significance(
						observed_statistic[comparison_effect].loc[line, genes], 
						null,
						tail=tail
					))
					significance[-1].rename(columns={
						col: "%s_%s" % (comparison_effect, col) 
						for col in significance[-1].columns
					}, inplace=True)

				out.append(pd.concat(significance, axis=1))
				out[-1]["cell_line_name"] = line
				out[-1]["total_readcount_bin"] = bin

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


################################################################
# P R O B A B I L I T I E S
################################################################


def smooth(x, sigma):
    '''smooth with a gaussian kernel'''


    kernel = np.exp(-.5 * np.arange(int(-4 * sigma), int(4 * sigma + 1), 1) ** 2 / sigma ** 2)
    kernel = kernel / sum(kernel)
    return np.convolve(x, kernel, 'same')


### Infer class probabilities
class MixFitEmpirical:
	'''
	fit a 1D mixture model with a set of known functions using E-M optimization
	
	'''
	def __init__(self, densities, data, initial_lambdas=None):
		'''
		Parameters:
			`densities`: iterable of normalized functions R^N -> P^N
			`data`: 1D array of points
			`initial_lambdas`: iterable with same length as densities giving initial guesses
				for size of each component. Must sum to 1.
		'''
		if initial_lambdas is None:
			initial_lambdas = [1.0/len(densities) for d in densities]
		if sum(pd.isnull(densities)) > 0:
			e = 'Error: %i null values in data\n%r' %(sum(pd.isnull(densities)), data)
			raise ValueError(e)
		self.lambdas = np.array(initial_lambdas)
		self.n = len(initial_lambdas)+1
		self.densities = list(densities)
		self.data = np.array(data)
		assert sum(initial_lambdas) == 1, 'Invalid mixing sizes'
		self.q = 1.0/(len(self.densities)) * np.ones((len(self.densities), len(self.data)))
		self.p = np.stack([d(self.data) for d in densities])
	

	def gauss(self, x):
		return norm.pdf(x, loc=self.mu, scale=self.sigma)

	
	def fit(self, tol=1e-7, maxit=1000, lambda_lock=False):
		'''
		Maximize likelihood. 
		Parameters:
			`tol` (`float`): consider converged when the fractional increase in likelihood 
				is less than this.
			`maxit` (`int`): maximum iterations to attempt for convergence.
			`lambda_lock` (`bool`): whether to update the total proportions
				of the observations estimated to be generated from each of the distributions.
				if `False`, will use the `initial_lambdas`. 
		'''
		self.likelihood = self.get_likelihood()
		for i in range(maxit):

			#E step
			for k in range(self.n-1):
				self.q[k, :] = self.lambdas[k]*self.p[k, :]
			if any(np.sum(self.q, axis=0) == 0):
				loc = np.argwhere(np.sum(self.q, axis=0) == 0).ravel()
				bad_points = self.data[loc]
				e = 'All component densities invalid for indices %r\ndata there: %r\ndensities there: %r)' %(
					loc, bad_points, [d(bad_points) for d in self.densities])
				raise ValueError(e)
			self.q /= np.sum(self.q, axis=0)

			#M step
			if not lambda_lock:
				self.lambdas = np.mean(self.q, axis=1)
				self.lambdas /= sum(self.lambdas)

			new_likelihood = self.get_likelihood()
			last_change = (new_likelihood - self.likelihood)/self.likelihood
			self.likelihood = new_likelihood
			if 0 <= last_change < tol:
				print("Converged at likelihood %f with %i iterations" %(new_likelihood, i))
				break
		if i >= maxit - 1:
			print("Reached max iterations %i with likelihood %f, last change %f" %(
				maxit, self.likelihood, last_change))
		
	def get_likelihood(self):
		out = np.mean(
			np.sum(
				self.q*np.log((self.p+1e-32)/(self.q+1e-32)),
				axis=1
			)
		)

		if np.isnan(out):
			e = "Null cost. %i nulls and %i negatives in q values, %i nulls and %i negatives in densities,\
%i nulls and %i negatives in p values\
			\n%r\n%r" %(
				np.sum(np.isnan(self.q)),
				np.sum(self.q < 0),
				np.sum(np.isnan(np.stack([d(self.data) for d in self.densities]))),
				np.sum(np.stack([d(self.data) for d in self.densities]) < 0),
				np.sum(np.isnan(self.p)),
				np.sum(self.p < 0),
				np.argwhere(np.isnan(self.p)),
				self.p.shape
				)
			raise ValueError(e)
		return out
	
	def full_density(self, points):
		out = 0*points
		for d,l in zip(self.densities, self.lambdas):
			out += l*d(points)
		return out
	
	def get_assignments(self, component):
		'''get probability that each data point was generated by the given component (int)'''
		return self.q[component]

	def component_probability(self, points, component):
		'''get probability that data at specified points belongs to the given component (int)'''
		return self.lambdas[component] * self.densities[component](points) / sum([
			l*d(points) for l, d in zip(self.lambdas, self.densities)
			])


class TailKernel:
	'''Used to set points at left and right extremes to fixed values'''
	def __init__(self,
				lower_bound=None, lower_value=None, upper_bound=None, upper_value=None):
		self.lower_bound = lower_bound
		self.lower_value = lower_value
		self.upper_bound = upper_bound
		self.upper_value = upper_value

	def apply(self, x, y):
		if self.lower_bound is not None:
			y[x < self.lower_bound] = self.lower_value
		if self.upper_bound is not None:
			y[x > self.upper_bound] =self.upper_value



				

def probability_2class(component0_points, component1_points, all_points,
			   smoothing='scott', p_smoothing=.15,
			   right_kernel_threshold=1, left_kernel_threshold=-3,
			   maxit=500, lambda_lock=False, mixfit_kwargs={}, 
			   kernel_kwargs=dict(lower_bound=-1.5, lower_value=1, upper_bound=.25, upper_value=0)):
	'''
	Estimates the distributions of component0_points and component1_points using a gaussian 
	kernel, then assigns each of all_points a probability of belonging to the component 
	distribution. Note that this is NOT a p-value: P(component1) = 1 - P(component0)
	'''
	#estimate density
	estimates = [
			gaussian_kde(component0_points, bw_method=smoothing),
			gaussian_kde(component1_points, bw_method=smoothing)
		]
	points = np.arange(min(all_points) - 4*p_smoothing, max(all_points) + 4*p_smoothing, .01)
	estimates = [e(points) for e in estimates]
	
	#this step copes with the fact that scipy's gaussian KDE often decays to true 0 in the far tails, leading to
	#undefined behavior in the mixture model
	if right_kernel_threshold is not None:
		estimates[0][np.logical_and(points > right_kernel_threshold, estimates[0] <1e-16)] = 1e-16
	if left_kernel_threshold is not None:
		estimates[1][np.logical_and(points < left_kernel_threshold, estimates[1] <1e-16)] = 1e-16

	#create density functions using interpolation (these are faster than calling KDE on points)
	densities = [interp1d(points, e) for e in estimates]
	
	#infer probabilities
	fitter = MixFitEmpirical(densities, all_points, **mixfit_kwargs)
	fitter.fit(maxit=maxit, lambda_lock=lambda_lock)

	#generate smoothed probability function
	p = fitter.component_probability(points, 1)
	probability_kernel = TailKernel(**kernel_kwargs)
	probability_kernel.apply(points, p)
	p = smooth(p, int(100*p_smoothing))
	d = interp1d(points, p)

	#return probabilities
	out = d(all_points)
	out[out > 1] =  1
	out[out < 0] = 0
	return out


def _check_controls(control_cols, control_matrix, gene_effect, label):
	var_name = '_'.join(label.split(' '))
	if control_cols is None and control_matrix is None:
		raise ValueError("One of `{0}` or `{0}_matrix` must be specified".format(var_name))
	if not (control_cols is None or control_matrix is None):
		raise ValueError("You can only specify one of `{0}` or `{0}_matrix` ".format(var_name))

	if control_matrix is None:
		missing = list(set(control_cols) - set(gene_effect.columns))
		if missing:
			warn("Not all %s found in the gene effect columns: %r" % (label, missing[:5]))
		control_cols = sorted(set(control_cols) & set(gene_effect.columns))
		if len(control_cols) < 10:
			raise ValueError("Less than 10 (%i) %s found in gene effect" % (len(control_cols), label))
		if len(control_cols) < 200:
			warn("Less than 200 (%i) %s found in gene effect, inference may be low quality" % 
				(len(control_cols), label))
		control_matrix = pd.DataFrame(False, index=gene_effect.index, columns=gene_effect.columns)
		control_matrix[control_cols] = True

	missing = set(gene_effect.index) - set(control_matrix.index)
	if len(missing) == len(gene_effect):
		raise ValueError("The index of `gene_effect` does not match `%s_matrix`." % (
			var_name))
	if len(missing):
		raise ValueError("Not all screens in `gene_effect` have entries in `%s_matrix`" % var_name)
	if not len(set(control_matrix.columns) & set(gene_effect.columns)):
		raise ValueError("None of the genes in `gene_effect` are in the columns of `%s_matrix`"
			% (var_name))
	control_matrix = control_matrix\
		.loc[gene_effect.index]\
		.reindex(columns=gene_effect.columns)\
		.fillna(False)
	if (control_matrix.sum(axis=1) < 200).any():
		warn("Less than 200 %s in some screens, inference may be low quality" % label)
	return control_matrix


def get_probability_dependent(gene_effect, 
	negative_controls=None, positive_controls=None, 
	negative_control_matrix=None, positive_control_matrix=None,
	**kwargs):
	'''
	Generates a matrix of the same dimensions as `gene_effect` where the values are probabilites
	that the corresponding `gene_effect` value was drawn from the distribution of 
	`positive_controls` rather than `negative_controls`.
	Parameters:
		`gene_effect` (`pandas.DataFrame`): Chronos gene effect estimates
		`negative_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as 
			negative controls (null distribution). `negative_controls` or 
			`negative_control_matrix` must be specified.
		`positive_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as
			positive_controls (high confidence that loss leads to loss of viability).
			One of `positive_controls` or `positive_control_matrix` must be specified.
		`negative_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a negative control
			in the cell line in question. This is useful when the controls differ betweeen different
			cell lines, such as when using unexpressed genes as the negative controls. 
		`positive_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a positive control
			in the cell line in question.  
		Other arguments passed to `probability_2class`
	Returns:
		`pandas.DataFrame` containing the probability estimates.
	'''
	

	positive_control_matrix = _check_controls(positive_controls, positive_control_matrix,
								gene_effect,  "positive controls")
	negative_control_matrix = _check_controls(negative_controls, negative_control_matrix,
								gene_effect,  "negative controls")

	# centers negative controls at 0. This is important for the tail kernels to behave correctly.
	gene_effect = gene_effect - np.nanmean(gene_effect.mask(~negative_control_matrix))

	def row_probability(x):
		return probability_2class(
			x[negative_control_matrix.loc[x.name]].dropna(), 
			x[positive_control_matrix.loc[x.name]].dropna(), 
			x.dropna(), 
			**kwargs
		)

	return pd.DataFrame({
		ind: pd.Series(row_probability(row), index=row.dropna().index)
		for ind, row in gene_effect.iterrows()
	}).reindex(index=gene_effect.columns).T


def get_fdr_from_probabilities(probabilities):
	'''Computes the (Bayesian) false discovery rates from the probability of dependency matrix 
	`probabilities` and returns it as a matrix. The FDR for an observation in a given
	cell line  is 1 - the mean probability of dependency for all observations
	in that cell line with equal or more negative gene effect than the observation.
	''' 
	out = {}
	for ind, row in probabilities.iterrows():
		row = row.sort_values(ascending=False)
		out[ind] = pd.Series(
			np.cumsum(1-row.fillna(0)) / np.cumsum(row.notnull()),
			index=row.index
		)
	return pd.DataFrame(out).reindex(index=probabilities.columns).T


def get_pvalue_dependent(gene_effect, negative_controls=None, negative_control_matrix=None):
	'''
	Generates a matrix of the same dimensions as `gene_effect` where the values are p-values
	for the null hypothesis that the true corresponding `gene_effect` value represented no viability
	effect against the alternative that there was loss of viability using `empirical_pvalue`.
	Parameters:
		`gene_effect` (`pandas.DataFrame`): Chronos gene effect estimates
		`negative_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as 
			negative controls (null distribution). `negative_controls` or 
			`negative_control_matrix` must be specified.
		`negative_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a negative control
			in the cell line in question. This is useful when the controls differ betweeen different
			cell lines, such as when using unexpressed genes as the negative controls. 
	Returns:
		`pandas.DataFrame` containing the p-value estimates.
	'''
	negative_control_matrix = _check_controls(negative_controls, negative_control_matrix,
		gene_effect, "negative_controls")
	return pd.DataFrame({
		ind: empirical_pvalue(row, row[negative_control_matrix.loc[ind]])
		for ind, row in gene_effect.iterrows()
	}).T


def get_fdr_from_pvalues(pvalues):
	'''Computes the Benjamini-Hochberg corrected p-values (frequentist false discovery rates)
	from the p-value matrix and returns it as a matrix. 
	FDRs are computed within individual cell lines (rows).
	''' 
	out = {}
	for ind, row in pvalues.iterrows():
		row = row.dropna()
		out[ind] = pd.Series(
			fdrcorrection(row, .05)[1],
			index=row.index
		)
	return pd.DataFrame(out).reindex(index=pvalues.columns).T