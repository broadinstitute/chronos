import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
import copy
from time import time
from datetime import timedelta, datetime
import h5py
from itertools import chain, combinations
import json
from warnings import warn

if tf.__version__ < "2":
	raise ImportError("Chronos requires tensorflow 2 or greater. Your version is %s." % tf.__version__)


class StdoutRedirector():
	'''
	simple widget to control print calls
	'''
	def __init__(self, output="stdout"):
		self.output = output
		if isinstance(output, str) and not (output == "stdout"):
			with open(output, "w") as f:
				current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				f.write("beginning log: %s\n" % current_time)

	def print(self, *string):
		if self.output == "stdout":
			print(*string)
		elif isinstance(self.output, str):
			with open(self.output, "a") as f:
				f.write('\t'.join(string) + "\n")
		elif self.output is None:
			pass
		else:
			raise ValueError("`output` for printing must be 'stdout', a file path, \
or `None`")


tf.compat.v1.disable_eager_execution()

'''
CHRONOS: population modeling of CRISPR readcount data
Joshua Dempster (dempster@broadinstitute.org)
The Broad Institute
'''

def powerset(iterable):
	'''
	taken from the itertools documentation
	powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
	'''
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def write_hdf5(df, filename):
	if os.path.exists(filename):
		os.remove(filename)
	dest = h5py.File(filename, 'w')

	try:
		dim_0 = [x.encode('utf8') for x in df.index]
		dim_1 = [x.encode('utf8') for x in df.columns]

		dest_dim_0 = dest.create_dataset('dim_0', track_times=False, data=dim_0)
		dest_dim_1 = dest.create_dataset('dim_1', track_times=False, data=dim_1)
		dest.create_dataset("data", track_times=False, data=df.values)
	finally:
		dest.close()


def read_hdf5(filename):
	src = h5py.File(filename, 'r')
	try:
		dim_0 = [x.decode('utf8') for x in src['dim_0']]
		dim_1 = [x.decode('utf8') for x in src['dim_1']]
		data = np.array(src['data'])

		return pd.DataFrame(index=dim_0, columns=dim_1, data=data)
	finally:
		src.close()


def check_if_unique(dictionary):
	'''
	checks if any of the (iterable of hashable) values in `dictionary` have overlapping 
	entries with any other value. Returns True if unique.
	'''
	keys = sorted(dictionary.keys())
	for i, key1 in enumerate(keys):
		for key2 in keys[i:]:
			overlap = set(dictionary[key1]) & set(dictionary[key2])
			if len(overlap):
				return False
	return True


def extract_last_reps(sequence_map):
		'''get the sequence IDs of replicates at their last measured timepoint'''
		rep_map = sequence_map[sequence_map.cell_line_name != 'pDNA']
		last_days = rep_map.groupby('cell_line_name').days.max()
		last_reps = rep_map[rep_map.days == last_days.loc[rep_map.cell_line_name].values].sequence_ID
		return last_reps


def check_inputs(readcounts=None, guide_gene_map=None, sequence_map=None):
	keys = None
	sequence_expected = set(['sequence_ID', 'cell_line_name', 'days', 'pDNA_batch'])
	guide_expected = set(['sgrna', 'gene'])

	for name, entry in zip(['readcounts', 'guide_gene_map', 'sequence_map'], [readcounts, guide_gene_map, sequence_map]):
		if entry is None:
			continue
		if not isinstance(entry, dict):
			raise ValueError("Expected dict, but received %r" %entry)
		if keys is None:
			keys = set(entry.keys())
		else:
			if not set(entry.keys()) == keys:
				raise ValueError("The keys for %s (%r) do not match the other keys found (%r)" % (name, keys, set(entry.keys())))
		for key, val in entry.items():
			if not isinstance(val, pd.DataFrame):
				raise ValueError('expected Pandas dataframe for %s[%r]' %(name, key))

			if name == 'readcounts':
				assert val.index.duplicated().sum() == 0, "duplicated index entries for readcounts %r" %key
				assert val.columns.duplicated().sum() == 0, "duplicated column names for readcounts %r" %key
				assert not val.isnull().all(axis=1).any(), \
						"All readcounts are null for one or more rows in %s, please drop them" % key
				assert not val.isnull().all(axis=0).any(),\
						 "All readcounts are null for one or more columns in %s, please drop them" % key

			elif name == 'guide_gene_map':
				assert not guide_expected - set(val.columns), \
						"not all expected columns %r found for guide-gene map for %s. Found %r" %(guide_expected, key, val.columns) 
				assert val.sgrna.duplicated().sum() == 0, \
					"duplicated sgRNAs for guide-gene map %r. Multiple gene alignments for sgRNAs are not supported." %key
				

			elif name == 'sequence_map':
				assert not sequence_expected - set(val.columns), \
						"not all expected columns %r found for sequence map for %s. Found %r" %(sequence_expected, key, val.columns)
				assert val.sequence_ID.duplicated().sum() == 0, "duplicated sequence IDs for sequence map %r" %key
				for batch in val.query('cell_line_name != "pDNA"').pDNA_batch.unique():
					assert batch in val.query('cell_line_name == "pDNA"').pDNA_batch.values, \
					"there are sequences with pDNA batch %s in library %s, but no pDNA measurements for that batch" %(batch, key)
				if val.days.max() > 50:
					warn("\t\t\t: many days (%1.2f) found for %s.\n\t\t\tThis may cause numerical issues in fitting the model.\n\
					Consider rescaling all days by a constant factor so the max is less than 50." % (val.days.max(), key))
				assert not (val.cell_line_name == 'pDNA').all(), "no late time points found for %s" % key
	
	for key in keys:
		if not readcounts is None and not sequence_map is None:
			assert not set(readcounts[key].index) ^ set(sequence_map[key].sequence_ID), \
				"\t\t\t mismatched sequence IDs between readcounts and sequence map for %r.\n\
				 Chronos expects `readcounts` to have guides as columns, sequence IDs as rows.\n\
				 Is your data transposed?" %key

		if not readcounts is None and not guide_gene_map is None:
			assert not set(readcounts[key].columns) ^ set(guide_gene_map[key].sgrna), \
				"mismatched map keys between readcounts and guide map for %s" % key

def check_negative_control_sgrnas(negative_control_sgrnas, readcounts):
	if not isinstance(negative_control_sgrnas, dict):
		raise ValueError("If supplied, negative_control_sgrnas must be a dict with libraries as keys")
	if set(negative_control_sgrnas.keys()) - set(readcounts.keys()):
		raise ValueError("Keys for negative_control_sgrnas not in readcounts keys")
	for key, val in negative_control_sgrnas.items():
		if (set(val) - set(readcounts[key].columns)):
			raise ValueError("Not all negative_control_sgrnas for %s present in readcounts" % key)


def filter_guides(guide_gene_map, max_guides=15):
	'''
	removes sgRNAs that target multiple genes, then genes that have less than two guides.
	Parameters:
		`guide_gene_map` (`pandas.DataFrame`): See Model.__init__ for formatting of guide_gene_map
	Returns:
		`pandas.DataFrame`: filtered guide_gene_map
	'''
	alignment_counts = guide_gene_map.groupby("sgrna").gene.count()
	guide_gene_map = guide_gene_map[guide_gene_map['sgrna'].isin(alignment_counts.loc[lambda x: x == 1].index)]
	guide_counts = guide_gene_map.groupby('gene')['sgrna'].count()
	guide_gene_map = guide_gene_map[guide_gene_map.gene.isin(guide_counts.loc[lambda x: (x > 1)& (x <= max_guides)].index)]
	return guide_gene_map


def calculate_fold_change(readcounts, sequence_map, rpm_normalize=True):
	'''
	Calculates fold change as the ratio of the late time points to pDNA
	Parameters:
		readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns
		sequence_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"
		rpm_normalize (`bool`): whether to normalize readcounts to have constant sum before calculating fold change.
								Should be true unless readcounts have been previously normalized.
	returns:
		fold_change (`pd.DataFrame`)
	'''
	check_inputs(readcounts={'default': readcounts}, sequence_map={'default': sequence_map})
	reps = sequence_map.query('cell_line_name != "pDNA"').sequence_ID
	pdna = sequence_map.query('cell_line_name == "pDNA"').sequence_ID
	rpm = readcounts
	if rpm_normalize:
		rpm = pd.DataFrame(
			(1e6 * readcounts.values.T / readcounts.sum(axis=1).values + 1).T,
			index=readcounts.index, columns=readcounts.columns
		)
	fc = rpm.loc[reps]
	pdna_reference = rpm.loc[pdna].groupby(sequence_map.set_index('sequence_ID')['pDNA_batch']).median()
	try:
		fc = pd.DataFrame(fc.values/pdna_reference.loc[sequence_map.set_index('sequence_ID').loc[reps, 'pDNA_batch']].values,
			index=fc.index, columns=fc.columns
			)
	except Exception as e:
		print(fc.iloc[:3, :3],'\n')
		print(pdna_reference[:3], '\n')
		print(reps[:3], '\n')
		print(sequence_map[:3], '\n')
		raise e
	errors = []
	# if np.sum(fc.values <= 0) > 0:
	#   errors.append("Fold change has zero or negative values:\n%r\n" % fc[fc <= 0].stack()[:10])
	# if (fc.min(axis=1) >= 1).any():
	#   errors.append("Fold change has no values less than 1 for replicates\n%r" % fc.min(axis=1).loc[lambda x: x>= 1])
	if errors:
		raise RuntimeError('\n'.join(errors))
	return fc

def moving_average(a, n=3) :
	ret = np.cumsum(a)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

def venter_mode(x, exclude=.1, windows=10):
	if (x > 0).sum() < 500:
		print("%s has less than 500 nonzero entries, returning median instead of mode" % x.name)
		return x.median()
	sort_x = x[x > 0].sort_values()
	window = len(sort_x)//windows
	rolling = moving_average(sort_x.values[window:] - sort_x.values[:-window], window)
	trim = int(round(exclude*len(sort_x)))
	interval_start = np.argmin(rolling[trim:-trim])+trim

	return sort_x.iloc[interval_start:interval_start+window].mean()

def normalize_readcounts(readcounts, negative_control_sgrnas=None, sequence_map=None):
	'''
	Normalizes readcounts
	Parameters:
		readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns
		negative_control_sgrnas (iterable or `None`): a set of CUTTING controls expected to produce no phenotype
		sequence_map (`pandas.DataFrame` or `None`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch".
													Must be supplied if negative_control_sgrnas are.
	returns:
		normnalized readcounts (`pd.DataFrame`): If readcounts has less than 2000 guides, this is just the readcount matrix
													with each sequenced entity (row) scaled to have the same MEDIAN.
												If more than 2000 guides but no negative controls are supplied,
													rows are instead scaled to have the same MODE in log space.
												If negative controls are supplied, pDNA entities are scaled to have the 
													same mode, and then late time points/replicates are scaled so that the 
													median of negative controls in each replicate matches the median
													of the negative controls in the corresponding pDNA batch
	'''
	if readcounts.shape[1] < 2000:
		warn("Readcounts has less than 2000 guides, using median normalization")
		return (readcounts.T / readcounts.T.median()).T
	elif negative_control_sgrnas is None:
		warn("No negative control sgRNAs supplied, aligning modes")
		return (
				readcounts.mean().mean()*(
				readcounts.T / 2.0**(np.log2(readcounts+1).T.apply(venter_mode))
			).T)
	if not len(negative_control_sgrnas):
		raise ValueError("set of negative_control_sgrnas is empty")
	if not len(set(negative_control_sgrnas) & set(readcounts.columns)):
		raise RuntimeError("None of the negative control sgrnas were found in the columns of readcounts")
	# Else, negative controls present:
	if len(set(negative_control_sgrnas) - set(readcounts.columns)):
		raise ValueError(
			"not all negative_control_sgrnas are present in the columns of readcounts"
		)
	if sequence_map is None:
		raise ValueError(
			"if negative_control_sgrnas are supplied, a sequence_map must be as well"
		)
	check_inputs(readcounts={'default': readcounts}, sequence_map={'default': sequence_map})
	logged = pd.DataFrame(np.log2(readcounts+1), 
						   index=readcounts.index, 
						   columns=readcounts.columns
						  )
	pdna_ids = sequence_map.query("cell_line_name == 'pDNA'")\
				.set_index("pDNA_batch").sequence_ID
	rep_ids = sequence_map.query("cell_line_name != 'pDNA'")\
				.set_index("pDNA_batch").sequence_ID
	modes = logged.loc[pdna_ids].T.apply(venter_mode)

	logged_pdna = (
		logged.loc[pdna_ids].T - modes + modes.median()
	).T
	
	pdna_ref = logged_pdna.groupby(pdna_ids.index).median()
	logged_reps = []
	for batch in rep_ids.index.unique():
		reps = rep_ids.loc[batch]
		if not isinstance(reps, pd.Series):
			reps = [reps]
		repbatch = logged.loc[reps]
		refbatch = pdna_ref.loc[batch, negative_control_sgrnas]
		shifts = repbatch[negative_control_sgrnas].median(axis=1) - refbatch.median()
		logged_reps.append(pd.DataFrame(repbatch.values-shifts.values.reshape((-1, 1)),
									   index=repbatch.index, columns=repbatch.columns))

	logged_reps = pd.concat(logged_reps, axis=0)
	
	
	assembled = pd.concat([logged_pdna, logged_reps], axis=0)
	
	return pd.DataFrame(2.0**assembled.values, index=assembled.index, 
						columns=assembled.columns)


def estimate_alpha(normalized_readcounts, negative_control_sgrnas, sequence_map, exclude_range=2.0,
	use_line_mean_as_reference=5):
	'''
	Estimates the overdispersion parameter alpha in the NB2 model: variance = mean * (1 + alpha * mean)
	Alpha is estimated using an "auxiliary" OLS model. The true mean of negative controls is assumed to
	be the median readcounts in the corresponding pDNA batch, and alpha is chosen to minimize the MSE
	of the variance (of counts in each replicate from the pDNA batch) from the expected trend. 
	Parameters:
		normalized_readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns, normalized
													with normalize_readcounts. Unnormalized counts will raise an error.
		negative_control_sgrnas (iterable or `None`): a set of CUTTING controls expected to produce no phenotype
		sequence_map (`pandas.DataFrame` or `None`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch".
													Must be supplied if negative_control_sgrnas are.
		exclude_range (`float`): some negative control sgRNAs are systematically different from pDNA in all late timepoints.
								We call this effect pDNA error. Including these sgRNAs will produce overly pessimistic
								estimates of alpha. Therefore, if the median of a negatve control sgRNA's reads across all 
								cell lines for a given pDNA batch deviates from the pDNA abundance by more than `exclude_range` 
								in log space, it will be excluded in that batch. Batches with fewer than 5 cell lines will not
								be filtered by this value.
		use_line_mean_as_reference (`int`): when at least this many lines are present for a pDNA batch, use the average
								of late time point sequences for that batch as the expected counts instead of pDNA counts.
								This gets around the problem of pDNA error.
	Returns:
		alphas (`pandas.Series`): a per-cell_line estimate of alpha.
	'''

	pdna_ids = sequence_map.query("cell_line_name == 'pDNA'")\
					.set_index("pDNA_batch").sequence_ID
	rep_ids = sequence_map.query("cell_line_name != 'pDNA'")\
				.set_index("pDNA_batch").sequence_ID
	expected = normalized_readcounts.loc[
		pdna_ids, negative_control_sgrnas
	].groupby(pdna_ids.index.values).median()
	# find negative controls which are consistently offset from pDNA in replicates, i.e.
	# the log of their median value in replicates is more than exclude_range away from the
	# the pDNA reference value. We can also check that the readcounts are normalized 
	# correctly.
	log_rep = pd.DataFrame(
		np.log2(normalized_readcounts.loc[rep_ids, negative_control_sgrnas].values),
		index=rep_ids, columns=negative_control_sgrnas
	)
	logexpected = np.log2(expected)
	normed_check = (log_rep.median(axis=1) - logexpected.median(axis=1).loc[rep_ids.index].values).abs()
	if (normed_check > .1).any():
		print(normed_check.sort_values(), '\n\n')

		raise ValueError("Normalized_readcounts have not been correctly normalized so \
the median value of negative control sgRNAs in each replicate matches the median value \
of the negative control sgRNAs in the corresponding pDNA samples. Run `normalize_readcounts` \
to obtain the correctly normalized readcount matrix before using this function.")
	log_rep_batch_median = log_rep.groupby(rep_ids.index.values).median()
	logexpected = logexpected.loc[log_rep_batch_median.index]	
	logdiff =  log_rep_batch_median - logexpected
	ndiff = (logdiff.abs() > exclude_range).sum(axis=1)
	warn("Between %i (batch=%r) and %i (batch=%r) negative control sgRNAs were found to be \
systematically over- or under-represented in the screens and excluded." % (
		ndiff.min(), ndiff.index[ndiff == ndiff.min()], 
		ndiff.max(), ndiff.index[ndiff == ndiff.max()]
	))

	mask = ~(logdiff.abs() < exclude_range)
	if (1-mask).sum(axis=1).min() < 100:
		raise ValueError("Fewer than 100 negative control sgRNAs remaining in one or more \
batches, too few to estimate overdispersion.")
	# if there are less than `use_line_mean_as_reference` lines in a batch, we don't want to 
	# exclude negative controls on the basis of being offset. 
	n_lines = sequence_map\
				.query("cell_line_name != 'pDNA'")\
				.groupby("pDNA_batch")\
				.cell_line_name\
				.nunique()

	too_few = n_lines.loc[lambda x: x < use_line_mean_as_reference].index
	mask.loc[too_few] = False
	# for batches with enough cell lines, use replicate median as reference instead
	for batch in n_lines.loc[lambda x: x >= use_line_mean_as_reference].index:
		expected.loc[batch] = normalized_readcounts.loc[
			rep_ids.loc[batch], 
			negative_control_sgrnas
		].median()
	expected.mask(mask, inplace=True)
	masked = normalized_readcounts.loc[rep_ids, negative_control_sgrnas].mask(
			expected.loc[rep_ids.index].isnull().values
		)

	varsum = ((masked - expected.loc[rep_ids.index].values) **2).sum(axis=1)
	expected_sum = expected.sum(axis=1).loc[rep_ids.index].values
	expected_squaresum = (expected**2.0).sum(axis=1).loc[rep_ids.index].values
	alpha = (varsum - expected_sum) / expected_squaresum

	return alpha

def nan_outgrowths(readcounts, sequence_map, guide_gene_map, absolute_cutoff=2, gap_cutoff=2,
				  rpm_normalize=False):
	'''
	NaNs readcounts in cases where all of the following are true:
		- The logfold change for the guide/replicate pair is greater than `absolute_cutoff`
		- The difference between the lfc for this pair and the next most positive pair for that gene and cell line is greater than 
			gap_cutoff. 
	Readcounts are mutated in place.
	Parameters:
		readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns
		sequence_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"
		guide_gene_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"

	'''
	
	check_inputs(readcounts={'default': readcounts}, sequence_map={'default': sequence_map},
					 guide_gene_map={'default': guide_gene_map})
	
	print('calculating LFC')
	fc = calculate_fold_change(readcounts, sequence_map, rpm_normalize)
	lfc = pd.DataFrame(
		np.log2(fc.values), index=fc.index,columns=fc.columns
	)

	print("stacking and annotating LFC")
	lfc_stack = lfc.copy()
	lfc_stack.index.name = "sequence_ID"
	lfc_stack.columns.name = "sgrna"
	
	lfc_stack = lfc_stack\
		.stack()\
		.reset_index()\
		.merge(sequence_map[["sequence_ID", "cell_line_name"]], how="left", on="sequence_ID")\
		.merge(guide_gene_map[["sgrna", "gene"]], how="left", on="sgrna")\
		.rename(columns={0: "LFC"})\
		.sort_values(["cell_line_name", "gene", "LFC"])\
		.reset_index(drop=True)

	print("finding group boundaries")
	gene_transitions = np.append(lfc_stack.gene.values[:-1] != lfc_stack.gene.values[1:], True)
	cell_transitions = np.append(lfc_stack.cell_line_name.values[:-1] != lfc_stack.cell_line_name.values[1:], True)
	transitions = gene_transitions | cell_transitions
	transition_indices = lfc_stack.index[transitions]

	print("removing cases with only one guide and replicate")
	number_lfcs = transition_indices[1:] - transition_indices[:-1]
	to_drop = transition_indices[[transition_indices[0] == lfc_stack.index[0]] + list(number_lfcs < 2)]
	lfc_stack.drop(to_drop, inplace=True)
	transition_indices = sorted(set(transition_indices) - set(to_drop))

	print("finding maximal values")
	maxima = lfc_stack.LFC[transition_indices].values
	second_maxima = lfc_stack.LFC[np.array(transition_indices).astype(np.int64)-1].values
	gaps = maxima - second_maxima


	print("making mask")
	lfc_stack["Mask"] = False
	bad_rows = np.array(transition_indices)[(maxima > absolute_cutoff) & (gaps > gap_cutoff)]
	print("found %i outgrowths, %1.1E of the total" % (len(bad_rows), len(bad_rows)/len(lfc_stack)))
	lfc_stack["Mask"].loc[bad_rows] = True

	print("pivoting mask")
	mask = pd.pivot(lfc_stack, values="Mask", index="sequence_ID", columns="sgrna")

	print("aligning_mask")
	mask = mask.reindex(index=readcounts.index, columns=readcounts.columns).fillna(False)

	print("NaNing")
	readcounts.mask(mask, inplace=True)


def load_saved_model(directory, readcounts=None, guide_gene_map=None, sequence_map=None, 
	negative_control_sgrnas=None, **kwargs):

	not_provided = [v is None for v in (readcounts, guide_gene_map, sequence_map)]
	if any(not_provided) and not all(not_provided):
		raise ValueError("if any of `readcounts`, `guide_gene_map`, or `sequence_map` are provided, \
all of them must be provided")
	required_files = [
		"cell_line_efficacy.csv", "cell_line_growth_rate.csv", "gene_effect.hdf5", "guide_efficacy.csv",
		"library_effect.csv", "parameters.json", "screen_delay.csv", "screen_excess_variance.csv",
		 "t0_offset.csv"
	]
	missing = set(required_files) - set(os.listdir(directory))
	if missing:
		raise IOError("directory %s is missing required files %r" % (directory, sorted(missing)))

	library_effect = pd.read_csv(os.path.join(directory, "library_effect.csv"), index_col=0)
	libraries = library_effect.columns

	with open(os.path.join(directory, "parameters.json")) as f:
		parameters = json.loads(f.read())
	usable_params = copy.copy(parameters)
	del usable_params['cost'], usable_params['full_cost']
	usable_params['smart_init'] = False
	usable_params.update(kwargs)

	to_normalize_readcounts = False
	if readcounts is None:
		to_normalize_readcounts = True
		candidate_readcounts = [s for s in os.listdir(directory) if s.endswith("_readcounts.hdf5")]
		if not len(candidate_readcounts):
			raise ValueError("No *_readcounts.hdf5 files found in %s. If no model inputs were saved, you must pass\
`readcounts`, `guide_gene_map`, and `sequence_map` dictionaries")
		try:
			readcounts = {library: read_hdf5(os.path.join(directory, library + "_readcounts.hdf5"))
						for library in libraries}
		except FileNotFoundError:
			raise FileNotFoundError("missing readcounts for some of the expected libraries %r" % sorted(libraries))
		try:
			guide_gene_map = {library: pd.read_csv(os.path.join(directory, library + "_guide_gene_map.csv"))
						for library in libraries}
		except FileNotFoundError:
			raise FileNotFoundError("missing guide_gene_map for some of the expected libraries %r" % sorted(libraries))
		try:
			sequence_map = {library: pd.read_csv(os.path.join(directory, library + "_sequence_map.csv"))
						for library in libraries}
		except FileNotFoundError:
			raise FileNotFoundError("missing sequence_map for some of the expected libraries %r" % sorted(libraries))

		negative_control_sgrnas = {}
		for library in libraries:
			try:
				negative_control_sgrnas[library] = pd.read_csv(os.path.join(directory, library + "_negative_control_sgrnas.csv")
					)['sgrna']
			except FileNotFoundError:
				pass

	model = Chronos(
			readcounts=readcounts,
			sequence_map=sequence_map,
			guide_gene_map=guide_gene_map,
			negative_control_sgrnas=negative_control_sgrnas,
			to_normalize_readcounts=False,
			**usable_params
		)

	model.printer.print("assigning trained parameters")
	model.printer.print("\tlibrary effect")
	model.library_effect = library_effect
	model.printer.print("\tgene effect")
	model.gene_effect = read_hdf5(os.path.join(directory, "gene_effect.hdf5"))
	model.printer.print("\tguide efficacy")
	model.guide_efficacy = pd.read_csv(os.path.join(directory, "guide_efficacy.csv"), index_col=0).iloc[:, 0]
	model.printer.print("\tcell efficacy")
	model.cell_efficacy = pd.read_csv(os.path.join(directory, "cell_line_efficacy.csv"), index_col=0)
	model.printer.print("\tcell growth rate")
	model.growth_rate = pd.read_csv(os.path.join(directory, "cell_line_growth_rate.csv"), index_col=0)
	model.printer.print("\tscreen excess variance")
	model.excess_variance = pd.read_csv(os.path.join(directory, "screen_excess_variance.csv"), index_col=0)
	model.printer.print("\tscreen delay")
	model.screen_delay = pd.read_csv(os.path.join(directory, "screen_delay.csv"), index_col=0).iloc[:,0]
	model.printer.print("\tt0 offset")
	model.t0_offset = pd.read_csv(os.path.join(directory, "t0_offset.csv"), index_col=0)

	model.printer.print("Complete.\nCost when saved: %f, cost now: %f\nFull cost when saved: %f, full cost now: %f" %(
		parameters['cost'], model.cost, parameters['full_cost'], model.full_cost
	))
	return model


##################################################################
#                M  O  D  E  L                                   #
##################################################################

class Chronos(object):
	'''
	Model class for inferring effect of gene knockout from readcount data. Takes in readcounts, mapping dataframes, 
	and hyperparameters at init, then is trained with `train`. 

	Note on axes:

	Replicates and cell lines are always the rows/major axis of all dataframes and tensors. Guides and genes are always the 
	columns/second axis. In cases where values vary per library, the object is a dict, and the library name is the key.

	Notes on attribute names:

	Attributes with single preceding underscores are tensorflow constants or tensorflow nodes, in analogy 
	with the idea of "private" attributes not meant to be interacted with directly. For tensorflow nodes,
	there is usually a defined class attribute with no underscore which runs the node and returns
	a pandas Series or DataFrame or dict of the same. 

	In other words `Chronos.v_a` (tensor)  --(tensorflow function)->  `Chronos._a` (tensor)  --(session run)->  `Chronos.a` (pandas object)

	Some intermediate tensorflow nodes do not have corresponding numpy/pandas attributes.

	Most parameters with a pandas interface can be set using the pandas interface. Do NOT try set tensorflow tensors directly - there
	are usually transformations Chronos expects, such as rescaling time values. Use the pandas interface, i.e.
	my_chronos_model.gene_effect = my_pandas_dataframe.

	Every set of parameters that are fit per-library are dicts. If `Chronos.v_a` is a dict, the subsequent attributes in the graph are 
	also dicts.


	Settable Attributes: these CAN be set manually to interrogate the model or for other advanced uses, but NOT RECOMMENDED. Most users 
	will just want to read them out after training.
		guide_efficacy (`pandas.Series`): estimated on-target KO efficacy of reagents, between 0 and 1
		cell_efficacy (`dict` of `pandas.Series`): estimated cell line KO efficacy per library, between 0 and 1
		growth_rate (`dict` of `pandas.Series`): relative growth rate of cell lines, positive float. 1 is the average of all lines in lbrary.
		gene_effect ('pandas.DataFrame'): cell line by gene matrix of inferred change in growth rate caused by gene knockout
		screen_delay (`pandas.Series`): per gene delay between infection and appearance of growth rate phenotype
		t0_offset (`dict` of 'pandas.Series'): per sgrna estimated log fold pDNA error, per library. This value is exponentiated and 
			mean-cented, then multiplied by the measured pDNA to infer the actual pDNA RPM of each guide.
			If there are fewer than 2 late time points, the mean of this value per gene is 0.
		days (`dict` of `pandas.Series`): number of days in culture for each replicate.
		learning_rate (`float`): current model learning rate. Will be overwritten when `train` is called.

	Unsettable (Calculated) Attributes:
		cost (`float`): the NB2 negative log-likelihood of the data under the current model, shifted to be 0 when the output RPM 
						perfectly matches the input RPM. Does not include regularization or terms involving only constants.
		cost_presum (`dict` of `pd.DataFrame`): the per-library, per-replicate, per-guide contribution to the cost.
		out (`dict` of `pd.DataFrame`): the per-library, per-replicate, per-guide model estimate of reads, unnormalized.
		predicted_readcounts (`dict` of `pandas.DataFrame`): `out` normalized so the sum of reads for each replicate is 1.
		efficacy (`pandas.DataFrame`): cell by guide efficacy matrix generated from the outer product of cell and guide efficacies
		t0 (`dict` of `pandas.DataFrame`): estimated t0 abundance of guides
		rpm (`dict` of `pandas.DataFrame`): the RPM of the measured readcounts / 1 million. Effectively a constant.
	'''

	default_timepoint_scale = .1 * np.log(2)
	default_cost_value = 0.67
	variable_max_value = 5
	persistent_handles = set([])
	def __init__(self, 
				 readcounts,
				 guide_gene_map,
				 sequence_map,
				 negative_control_sgrnas={},

				 gene_effect_hierarchical=.1,
				 gene_effect_smoothing=1.5,
				 kernel_width=50,
				 gene_effect_L1=0.1,
				 gene_effect_L2=0,
				 offset_reg=1,
				 excess_variance=0.02,
				 guide_efficacy_reg=.01,
				 library_batch_reg=.1,
				
				 growth_rate_reg=0.01,
				 smart_init=True,
				 pretrained=False,
				 cell_efficacy_guide_quantile=0.02,
				 initial_screen_delay=3,
				 scale_cost=0.67,
				 max_learning_rate=.04,
				 dtype=tf.double,
				 verify_integrity=True, 
				 log_dir=None,
				 to_normalize_readcounts=True,
				 use_line_mean_as_reference=5,
				 print_to="stdout"
				):
		'''
		Parameters:
			readcounts (`dict` of `pandas.DataFrame`): Values are matrices with sequenced entities on rows, 
				guides as column headers, and total readcounts for the guide in the replicate as entries. There should be at least one key 
				for each library, but the user can also make separate individual datasets according to some other condition,
				such as screening site.
			sequence_map (`dict` of `pandas.DataFrame`): Keys must match the keys of readcounts. Values are tables with the columns: 
				sequence_ID: matches a row index in the corresponding readcounts matrix. Should uniquely identify a combination of
							 cell line, replicate, and sequence passage.
				cell_line: name of corresponding cell line. 'pDNA' if this is a plasmid DNA or initial count measurement.
				days: estimate number of cell days from infection when readcounts were performed. Plasmid DNA entries should be 0.
				pDNA_batch: Unique identifier for associating readcounts to time 0 readcounts. 
			guide_gene_map (`dict` of `pandas.DataFrame`): Values are tables with the columns:
				sgrna: guide sequence or unique guide identifier
				gene: gene mapped to by guide. Genes should follow consistent naming conventions between libraries
			negative_control_sgrnas (`dict` of iterables): for each library, a list of CUTTING sgRNAs which are expected
				to produce no growth phenotype, for example intergenic sgRNAs or those targeting olfactory receptors. Missing keys
				will cause Chronos not to produce overdispersion estimates and to default to mode or median
				readcount normalization for that library.

			gene_effect_hierarchical (`float`): regularization of individual gene effect scores towards the mean across cell lines
			gene_effect_smoothing (`float`): regularization of individual gene scores towards mean after Gaussian kernel convolution.
				This removes trends rather than individual outliers.
			kernel_width (`float`): width (SD) of the Gaussian kernel for the smoothing regularization
			gene_effect_L1 (`float`): regularization of gene effect CELL LINE MEAN towards zero with L1 penalty
			gene_effect_L2 (`float`): regularization of individual gene scores towards zero with L2 penalty
			offset_reg (`float`): regularization of pDNA error
			growth_rate_reg (`float`): regularization of the negative log of the relative growth rate
			excess_variance (`float` or `dict`): measure of Negative Binomial 2 overdispersion for the cost function, 
								overall or per replicate/library. If this is a float and negative_control_sgRNAs 
								are passed, this will be overwritten by the estimates produced by `estimate_alpha`.
			guide_efficacy_reg (`float`): regularization of guide efficacy towards 1.
			library_batch_reg (`float`): regularization of gene means within libraries towards the global gene mean.
			smart_init (`bool`): if True (default) the model initializes cell efficacy and gene effect by using estimates
				based on the fold change of the latest available time points. If this parameter is False, cell_line_efficacy
				will be 1 for all cell lines!
			pretrained (`bool`): whether the model is being initialized from a pretrained state using orthogonal data.
			cell_efficacy_guide_quantile (`float`): quantile of guides to use to estimate cell screen efficacy. Between 0 and 0.5.
			initial_screen_delay (`float`): how long after infection before growth phenotype kicks in, in days. If there are fewer than
								3 late timepoints this initial value will be left unchanged.
			max_learning_rate (`float`): passed to AdamOptimizer after initial burn-in period during training
			cell_efficacy_init (`bool`): whether to initialize cell efficacies using the fold change of the most depleted guides 
								at the last timepoint
			
			
			dtype (`tensorflow.double` or `tensorflow.float`): numerical precision of the computation. Strongly recommend to leave this 
								unchanged.
			verify_integrity (`bool`): whether to check each itnermediate tensor computed by Chronos for innappropriate values
			log_dir (`path` or None): if provided, location where Tensorboard snapshots will be saved
			scale_cost (`bool`): The likelihood cost will be scaled to always be initially this value (default 0.67) for all data. 
								This encourages more consistent behavior across datasets when leaving the other regularization hyperparameters 
								constant. Pass 0, False, or None to avoid cost scaling.
			to_normalize_readcounts (`bool`): If true, the readcounts will be normalized. if negative_control_sgRNAs are provided,
								Chronos will normalize such that the median log reads of negative controls in each replicate match
								the median in the pDNA batch. 
			use_line_mean_as_reference (`int`): passed to `estimate_alpha`
			print_to (`str` or `None`): where to print ordinary messages from Chronos. Default is `stdout`. Pass a file path to print
								to the file or `None` to skip these messages.

		Attributes:
			Attributes beginning wit "v_" are tensorflow variables, and attributes beginning with _ are 
			tensorflow nodes. There should be no need to access these directly. All the relevant tensorflow
			nodes are exposed as properties which return properly indexed Pandas objects. 
			
		Properties (type `help(Chronos.<property>) to learn more):
			cell_efficacy
			cost
			cost_presum
			days
			estimated_fold_change
			full_cost
			gene_effect
			guide_efficacy
			growth_rate
			efficacy
			excess_variance
			learning_rate
			library_means
			library_effect
			mask
			measured_t0
			normalized_readcounts
			predicted_readcounts_scaled
			predicted_readcounts
			screen_delay
			t0
			t0_core
			t0_offset
		'''


		###########################    I N I T I A L      C  H  E  C  K  S  ############################
		self.printer = StdoutRedirector(print_to)

		check_inputs(readcounts=readcounts, sequence_map=sequence_map, guide_gene_map=guide_gene_map)
		check_negative_control_sgrnas(negative_control_sgrnas, readcounts)
		self.guide_gene_map = guide_gene_map


		sequence_map = self._make_pdna_unique(sequence_map, readcounts)
		if to_normalize_readcounts:
			self.printer.print("normalizing readcounts")
			readcounts = {key: normalize_readcounts(val, negative_control_sgrnas.get(key), sequence_map[key])
						for key, val in readcounts.items()}
		self.sequence_map = sequence_map
		self.readcounts = readcounts
		self.negative_control_sgrnas = negative_control_sgrnas

		excess_variance = self._check_excess_variance(excess_variance, readcounts, sequence_map)
		self.np_dtype = {tf.double: np.float64, tf.float32: np.float32}[dtype]
		self.keys = list(readcounts.keys())
		if scale_cost:
			try:
				scale_cost = float(scale_cost)
				assert 0 < scale_cost, "scale_cost must be positive"
			except:
				raise ValueError("scale_cost must be None, False, or a semi-positive number")

		self.zero = tf.constant(0, dtype)


		####################    C  R  E  A  T  E       M  A  P  P  I  N  G  S   ########################

		#genes is a library dict giving the set of genes overed by the given library.
		#all_guides and all_genes give the order for tensors combined over libraries (the union of the libraries).
		#intersecting_genes is the intersection of the genes in each library.
		#guide_map is a library dict. For each library, it contains keys gather_index_inner and gather_index_outer.
		#Here, gather_index_outer is the array of integer indices that need to be selected from all_guides to get
		#the guides in the guide_map for the library (in order). gather_index_inner is the array of integer indices 
		#that need to be
		#selected from all_genes to get the corresponding gene for each guide in that library.
		#unified_guide_map is similar, but contains the mapping of all_genes to all_guides (ignoring library).
		#column_map is also a library dict, giving the string indices (sgrna) for guides in the order they appear.
		#in the readcounts
		(self.genes, self.all_guides, self.all_genes, self.intersecting_genes,
			self.guide_map, self.unified_guide_map, self.column_map
			) = self._get_column_attributes(readcounts, guide_gene_map)

		#cells is a library dict giving the set of cell lines overed by the given library.
		#cell_indices is a library dict of integers giving the indices in all_cells that should be selected to get
		#rows matching self.cells. This is used by copy number correction.
		#all_sequences and all_cells give the order for tensors combined over libraries.
		#replicate_map is a library dict. For each library, it contains keys gather_index_inner and gather_index_outer.
		#Here, gather_index_outer is the array of integer indices that need to be selected from all_sequences to get
		#the sequences in the sequence map for that library (in order). gather_index_inner is the array of integer indices 
		#that need to be selected from all_cells to get the corresponding cell line for each sequence in that library.
		#index_map is the string indices (sequence_IDs) corresponding to the given row in the library.
		#batch_map is similar to replicate_map, but maps pDNA batch rows to the corresponding late timepoints in each library.
		(self.cells, self.all_sequences, \
			self.all_cells, self.pDNA_unique, self.cell_indices, self.replicate_map, self.index_map, 
			self.batch_map
			) = self._get_row_attributes(readcounts, sequence_map)


		##################    A  S  S  I  G  N       C  O  N  S  T  A  N  T  S   #######################

		self.printer.print('\n\nassigning float constants')
		self.guide_efficacy_reg = float(guide_efficacy_reg)
		self.gene_effect_L1 = float(gene_effect_L1)
		self.gene_effect_L2 = float(gene_effect_L2)
		self._private_gene_effect_hierarchical = float(gene_effect_hierarchical)
		self._gene_effect_hierarchical = tf.compat.v1.placeholder(dtype, shape=())
		self.growth_rate_reg = float(growth_rate_reg)
		self.offset_reg = float(offset_reg)
		self.gene_effect_smoothing = float(gene_effect_smoothing)
		self.kernel_width = float(kernel_width)
		self.cell_efficacy_guide_quantile = float(cell_efficacy_guide_quantile)
		self.library_batch_reg = float(library_batch_reg)
		self.use_line_mean_as_reference = use_line_mean_as_reference

		if not 0 < self.cell_efficacy_guide_quantile < .5:
			raise ValueError("cell_efficacy_guide_quantile should be greater than 0 and less than 0.5")

		self.nguides, self.ngenes, self.nlines, self.nsequences = (
			len(self.all_guides), len(self.all_genes), len(self.all_cells), len(self.all_sequences)
		)
		#the alpha parameter of an NB2 model for readcount noise, one value per sequence, dict by library
		#behavior depends on whether negative_control_sgrnas are supplied for each library.
		excess_variance = self._estimate_excess_variance(
			excess_variance, readcounts, negative_control_sgrnas, sequence_map, use_line_mean_as_reference
		)
		self._excess_variance = self._get_excess_variance_tf(excess_variance)

		self.median_timepoint_counts = self._summarize_timepoint(sequence_map, np.median)

		self.median_guide_counts = self._get_median_guide_counts(self.unified_guide_map)

		#set up the graph
		self._initialize_graph(max_learning_rate, dtype)
		#the libraries may cover different genes, so gene effect estimates for a cell line in one
		#library may not be meaningful if there are no guides. The mask NAs that value for both
		#optimization and when the model reports values
		self._gene_effect_mask, self.mask_count = self._get_gene_effect_mask(readcounts, sequence_map, 
														guide_gene_map, dtype)
		#tensor days are multipled by the default_timepoint_scale to reduce the risk of over/underflow
		self._days = self._get_days(sequence_map, dtype)
		#readcounts are aligned to the internal 
		self._normalized_readcounts, self._mask = self._get_late_tf_timepoints(readcounts, dtype)
		#The measured pDNA per batch as a persistent tensor, normed to have total 1, 
		#and the pdna_scale, a per-sequence number defined as the sum of all reads in the pDNA batch times 
		# the ratio of the median reads in the readcounts for the given sequence to the median reads in the 
		# pDNA batch. pdna_scale will scale the model's readcount estimates so that they are aligned with
		# the observed readcounts. The observed readcounts are no longer RPM, because this would make the
		# inferred excess_variance incorrect; hence the need to have a specific number to scale the output to.
		self._measured_t0, self._pdna_scale = self._get_tf_measured_t0(readcounts, sequence_map, dtype)

		self._pretrained = pretrained
		self._is_model_loaded = False
		

		##################    C  R  E  A  T  E       V  A  R  I  A  B  L  E  S   #######################

		self.printer.print('\n\nBuilding variables')

		(
				self.v_t0, self._t0_core, 
				self._t0, self._t0_offset, self._grouped_t0_offset,
				self._grouped_offset_denom, self._combined_t0_offset,
				self._masked_t0, self._t0_mask_input
		) = self._get_t0_tf_variables(self._measured_t0, dtype)

		# the tensorflow variable (1, len(self.all_guides)+1), and the tranformation of it to the interval [0, 1]
		(self.v_guide_efficacy, self._guide_efficacy_mask_input, self._guide_efficacy) = self._get_tf_guide_efficacy(dtype)
		# the relative growth rate for cell lines in each library (library dict).
		#Growth_rate is constrained to be positive and scaled to have mean 1.
		(self.v_growth_rate, self._growth_rate, self._line_presence_boolean) = self._get_tf_growth_rate(dtype)
		# cell_efficacy is constrained between 0 and 1. Currently estimated at
		#initialization and not updated during training. This value is estimated per cell line
		#in each library (library dict)
		(self.v_cell_efficacy, self._cell_efficacy) = self._get_tf_cell_efficacy(dtype)
		# screen delay: currently not updated during training. This is a per-library delay between 
		# knockout and viability impact. 
		(self.v_screen_delay, self._screen_delay) = self._get_tf_screen_delay(initial_screen_delay, dtype)
		# Gene effect is comprised of two parts: a per gene mean effect which is constant over all cell lines,
		# and normaized by the L1 penalty to have mean near 0; and a per line, per gene deviation from the mean.
		# This is the value penalized by the smoothing and hierarchical penalties.
		# _true_residue is this deviation after constraining v_residue to have mean 0. 
		# _combined_gene_effect is the sum of v_mean_effect and _true_residue. This is the tensor 
		# accessed by the attribute Chronos.gene_effect.
		(self.v_mean_effect, self.v_residue, self._residue, self._true_residue, 
			self._combined_gene_effect, self.v_library_effect, self._library_effect
		) = self._get_tf_gene_effect(dtype)


		#############################    C  O  R  E      M  O  D  E  L    ##############################

		self.printer.print("\n\nConnecting graph nodes in model")
		# _effective_days: _days - _screen_delay clipped to be semipositive. Library dict per cell line
		self._effective_days = self._get_effect_days(self._screen_delay, self._days)
		#_gene_effect_growth: 
		self._gene_effect_growth = self._get_gene_effect_growth(self._combined_gene_effect, self._growth_rate,
			self._library_effect)
		# _efficacy: the outer product of the cell line and guide efficacies for each library. _selected_efficacies
		# is this object broadcast so there is one row per non-pDNA sequence for the library.  
		self._efficacy, self._selected_efficacies = self._get_combined_efficacy(self._cell_efficacy,self. _guide_efficacy)
		# in _growth, _gene_effect_growth is first broadcast to the guide and sequence level, then the exponent taken
		# (library dict).
		# _change is the espected fold change according to the Chronos model for each guide in each sequence (library dict).
		self._growth, self._change = self._get_growth_and_fold_change(self._gene_effect_growth, self._effective_days, 
																		self._selected_efficacies)
		#_out is the library dict of relative expected abundance, obtained by multiplying _change by _t0.
		# this is normalized to have sum 1, then multiplied by _pdna_scale to get the absolute expected reads.
		self._predicted_readcounts_unscaled, self._predicted_readcounts = self._get_abundance_estimates(self._t0, self._change)

		init_op = tf.compat.v1.global_variables_initializer()
		self.printer.print('initializing precost variables')
		self.sess.run(init_op)

		#####################################    C  O  S  T    #########################################

		self.printer.print("\n\nBuilding all costs")

		self._total_guide_reg_cost = self._get_guide_regularization_alt(self._guide_efficacy, dtype)

		self._smoothed_presum = self._get_smoothed_ge_regularization(self.v_mean_effect, self._true_residue, kernel_width, dtype)

		self._t0_cost = self._get_t0_regularization(self._t0_offset)
			
		self._cost_presum, self._cost_constant, self._cost, self._scale = self._get_nb2_cost(
			self._excess_variance, self._predicted_readcounts, 
			self._normalized_readcounts, self._mask, dtype
		)

		self._library_means, self._library_batch_cost = self._get_library_batch_reg(
			self._true_residue, self.library_batch_reg, dtype
		)

		self.run_dict.update({self._scale: 1.0})

		self._full_cost = self._get_full_cost(dtype)


		#########################    F  I  N  A  L  I  Z  I  N  G    ###################################

		self.printer.print('\nCreating optimizer')       
		self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._learning_rate, name="Adam")


		self.default_var_list = [
				self.v_mean_effect,
				self.v_residue, 
				self.v_guide_efficacy,
				self.v_t0,
				self.v_growth_rate,
				self.v_library_effect
				]
		if self.median_guide_counts <= 2:
			self.printer.print("Two or fewer guides for most genes, guide efficacy will not be trained")
			self.default_var_list.remove(self.v_guide_efficacy)

		self._step = self.optimizer.minimize(self._full_cost, var_list=self.default_var_list)
		self._ge_only_step = self.optimizer.minimize(self._full_cost, var_list=[self.v_mean_effect, self.v_residue])
		self._loaded_model_step = self.optimizer.minimize(self._full_cost, var_list=[self.v_residue, 
			self.v_growth_rate])        
		self._merged = tf.compat.v1.summary.merge_all()

		if log_dir is not None:
			self.printer.print("\tcreating log at %s" %log_dir)
			if os.path.isdir(log_dir):
				shutil.rmtree(log_dir)
			os.mkdir(log_dir)
			self.log_dir = log_dir
			self.writer = tf.compat.v1.summary.FileWriter(log_dir, self.sess.graph)
		
		init_op = tf.compat.v1.global_variables_initializer()
		self.printer.print('initializing rest of graph')
		self.sess.run(init_op)

		if scale_cost:
			denom = self.cost
			self.run_dict.update({self._scale: scale_cost/denom})

		if smart_init:
			self.printer.print("estimating initial screen efficacy and gene effect")
			self.smart_initialize(readcounts, sequence_map, guide_gene_map, cell_efficacy_guide_quantile, negative_control_sgrnas,
				initial_screen_delay)

		if verify_integrity:
			self.printer.print("\tverifying graph integrity")
			self.nan_check()

		self.epoch = 0

		if self._pretrained:
			self.printer.print('waiting for user to load model')
		else:
			self.printer.print('ready to train')



	################################################################################################
	##############   I N I T I A L I Z A T I O N    M  E  T  H  O  D  S    #########################
	################################################################################################


	def get_persistent_input(self, dtype, data, name=''):
		with tf.compat.v1.name_scope(name):
			placeholder = tf.compat.v1.placeholder(dtype=dtype, shape=data.shape, name="placeholder")
			# Persistent tensor to hold the data in tensorflow. Helpful because TF doesn't allow 
			# graph definitions larger than 2GB (so can't use constants), and passing the feed dict each time is slow.
			# This feature is poorly documented, but the handle seems to refer not to a tensor but rather a tensor "state" -
			# the state of a placeholder that's been passed the feed dict. This is what persists. Annoyingly, it then becomes
			# impossible to track the shape of the tensor.
			state_handle = self.sess.run(tf.compat.v1.get_session_handle(placeholder), {placeholder: data})
			# why TF's persistence requires two handles, I don't know. But it does.
			tensor_handle, data = tf.compat.v1.get_session_tensor(state_handle.handle, dtype=dtype, name="handle")
			self.run_dict[tensor_handle] = state_handle.handle
			self.persistent_handles.add(state_handle.handle)
		return data


###########################    I N I T I A L      C  H  E  C  K  S  ############################

	def _make_pdna_unique(self, sequence_map, readcounts):
		#guarantee unique pDNA batches
		if check_if_unique({key: val['pDNA_batch'] for key, val in sequence_map.items()}):
			return sequence_map

		sequence_map = {key: val.query('sequence_ID in %r' % list(readcounts[key].index)) for key, val in sequence_map.items()}
		for key, val in sequence_map.items():
			val['pDNA_batch'] = val['pDNA_batch'].apply(lambda s: '%s_%s' % (key, s))
		return sequence_map


	def _check_excess_variance(self, excess_variance, readcounts, sequence_map):
		if not isinstance(excess_variance, dict):
			try:
				excess_variance = float(excess_variance) 
			except ValueError:
				raise ValueError("if provided, excess_variance must be a dict of pd.Series per library or a float")
		else:
			for key, val in excess_variance.items():
				assert key in readcounts, "excess_variance key %s not found in the rest of the data" % key
				assert isinstance(val, pd.Series), \
					"the excess_variance values provided for the different datasets must be pandas.Series objects, not\n%r" % val
				diff = set(val.index) ^ set(sequence_map[key].query("cell_line_name != 'pDNA'").sequence_ID)
				assert len(diff)==0, \
					"difference between index values\n%r\nfor excess_variance and replicates found in %s" % (diff, key)
		return excess_variance


####################    C  R  E  A  T  E       M  A  P  P  I  N  G  S   ########################

	def make_map(melted_map, outer_list, inner_list, dtype=np.float64):
		'''
		takes a sorted list of indices, targets, and a pd.Series that maps between them and 
		recomputes the mapping between them as two arrays of integer indices suitable for gather 
		function calls. 
		Specifically:
			outer_list[gather_ind_outer[i]] = melted_map.index[i]
			inner_list[gather_ind_inner[i]] = melted_map.values[i]
			melted_map.index[reverse_map_outer[i]] = outer_list[i]
		The mapping can only include a subset of either the outer or inner list and vice versa.
		The mapping's indices must be unique.
		
		'''
		melted_map = melted_map[melted_map.index.isin(outer_list) & melted_map.isin(inner_list)]
		outer_array = np.array(outer_list)
		gather_outer = np.searchsorted(outer_array, melted_map.index).astype(int)
		inner_array = np.array(inner_list)
		gather_inner = np.searchsorted(inner_array, melted_map.values).astype(int)
		gather_outer_reversed = np.arange(len(gather_outer))[np.argsort(gather_outer)]
		reverse_map = pd.Series(np.arange(len(gather_outer)), index=gather_outer)
		args = { 
			'gather_ind_inner': gather_inner,
			'labels_inner': inner_array[gather_inner], 
			'gather_ind_outer': gather_outer,
			'labels_outer': outer_array[gather_outer],
			'reverse_map_outer': reverse_map

		}
		return args


	def _get_column_attributes(self, readcounts, guide_gene_map):
		self.printer.print('\n\nFinding all unique guides and genes')
		#guarantees the same sequence of guides and genes within each library
		guides = {key: val.columns for key, val in readcounts.items()}
		genes = {key: val.set_index('sgrna').loc[guides[key], 'gene'] for key, val in guide_gene_map.items()}
		all_guides = sorted(set.union(*[set(v) for v in guides.values()]))
		all_genes = sorted(set.union(*[set(v.values) for v in genes.values()]))
		intersecting_genes = sorted(set.intersection(*[set(v.values) for v in genes.values()]))
		for key in self.keys:
			self.printer.print("found %i unique guides and %i unique genes in %s" %(
				len(set(guides[key])), len(set(genes[key])), key
				))
		self.printer.print("found %i unique guides and %i unique genes overall" %(len(all_guides), len(all_genes)))
		self.printer.print('\nfinding guide-gene mapping indices')
		#in guide_map, gather_index_inner gives the index of the targeted gene in all_genes
		#gather_index_outer gives the index of the sgrna in all_guides
		#concretely, for any tensor with guides as columns, the ith column is the sgrna 
		#       all_guides[ guide_map[key]['gather_index_outer'][i] ], 
		#and the gene it targets is 
		#       all_genes[ guide_map[key]['gather_index_inner'][i] ] 
		guide_map = {key: 
				Chronos.make_map(guide_gene_map[key][['sgrna', 'gene']].set_index('sgrna').iloc[:, 0],
				 all_guides, all_genes, self.np_dtype)
				for key in self.keys}
		column_map = {key: np.array(all_guides)[guide_map[key]['gather_ind_outer']]
								for key in self.keys}

		unified = pd.concat([
				guide_gene_map[key][['sgrna', 'gene']]
				for key in self.keys
			], ignore_index=True)\
			.drop_duplicates(subset=['sgrna', 'gene'])
		duplicates = unified[unified.sgrna.duplicated()].sgrna
		if len(duplicates):
			raise ValueError("Inconsistent gene annotations seen for the same sgrna in different libraries.\n%r" % 
				unified[unified.sgrna.isin(duplicates)].sort_values("sgrna"))
		unified_guide_map = pd.DataFrame(Chronos.make_map(unified.set_index('sgrna').loc[all_guides, 'gene'], 
			all_guides, all_genes))

		return genes, all_guides, all_genes, intersecting_genes, guide_map, unified_guide_map, column_map


	def _get_row_attributes(self, readcounts, sequence_map):
		self.printer.print('\nfinding all unique sequenced replicates, cell lines, and pDNA batches')
		#guarantees the same sequence of sequence_IDs and cell lines within each library.
		sequences = {key: val[val.cell_line_name != 'pDNA'].sequence_ID for key, val in sequence_map.items()}
		pDNA_batches = {key: list(val[val.cell_line_name != 'pDNA'].pDNA_batch.values)
								for key, val in sequence_map.items()}
		pDNA_unique = {key: sorted(set(val)) for key, val in pDNA_batches.items()}
		cells = {key: val[val.cell_line_name != 'pDNA']['cell_line_name'].unique() for key, val in sequence_map.items()}
		all_sequences = sorted(set.union(*tuple([set(v.values) for v in sequences.values()])))
		all_cells = sorted(set.union(*tuple([set(v) for v in cells.values()])))
		#This is necessary to consume copy number provided for only the cell-guide blocks present in each library
		cell_indices = {key: [all_cells.index(s) for s in v] 
									for key, v in cells.items()}

		assert len(all_sequences) == sum([len(val) for val in sequences.values()]
			), "sequence IDs must be unique among all datasets"
		for key in self.keys:
			self.printer.print("found %i unique sequences (excluding pDNA) and %i unique cell lines in %s" %(
				len(set(sequences[key])), len(set(cells[key])), key
				))
		self.printer.print("found %i unique replicates and %i unique cell lines overall" %(len(all_sequences), len(all_cells)))

		self.printer.print('\nfinding replicate-cell line mappings indices')
		# in replicate_map, gather_index_inner gives the index of the replicate's cell line in all_cells
		# gather_index_outer gives the index of the replicate in all_sequences
		replicate_map = {key: 
				Chronos.make_map(sequence_map[key][['sequence_ID', 'cell_line_name']].set_index('sequence_ID').iloc[:, 0],
				 all_sequences, all_cells, self.np_dtype)
				for key in self.keys}
		index_map = {key: np.array(all_sequences)[replicate_map[key]['gather_ind_outer']]
								for key in self.keys}

		self.printer.print('\nfinding replicate-pDNA mappings indices')
		batch_map = {key: 
				Chronos.make_map(sequence_map[key][['sequence_ID', 'pDNA_batch']].set_index('sequence_ID').iloc[:, 0],
				 all_sequences, pDNA_unique[key], self.np_dtype)
				for key in self.keys}

		return cells, all_sequences, all_cells, pDNA_unique, cell_indices, replicate_map, index_map, batch_map


##################    A  S  S  I  G  N       C  O  N  S  T  A  N  T  S   #######################


	def _estimate_excess_variance(self, excess_variance, readcounts, negative_control_sgrnas, sequence_map,
			use_line_mean_as_reference):
		self.printer.print('Estimating or aligning variances')
		if not isinstance(excess_variance, dict):
			prior_variance = excess_variance
			excess_variance = {}
		for key in self.keys:
			if not (negative_control_sgrnas.get(key) is None) and not key in excess_variance:
				self.printer.print('\tEstimating excess variance (alpha) for %s' % key)
				excess_variance[key] = estimate_alpha(
						readcounts[key], negative_control_sgrnas[key], sequence_map[key],
						use_line_mean_as_reference=use_line_mean_as_reference
					)[self.index_map[key]]
			elif not key in excess_variance:
				excess_variance[key] = pd.Series(prior_variance, index=self.index_map[key])
		return excess_variance

	def _get_excess_variance_tf(self, excess_variance):
		self.printer.print("Creating excess variance tensors")
		_excess_variance = {}
		with tf.compat.v1.name_scope("excess_variance"):
			for key in self.keys:
				try:
					_excess_variance[key] = tf.Variable(
						excess_variance[key][self.index_map[key]].values.reshape((-1, 1)),
						name=key
					)
				except IndexError:
					raise IndexError("difference between index values for excess_variance and replicates found in %s" % key)
				except TypeError:
					_excess_variance[key] = tf.Variable(excess_variance * np.ones(shape=(len(self.index_map[key]), 1)))
				self.printer.print("\tCreated excess variance tensor for %s with shape %r" % (key, _excess_variance[key].get_shape().as_list()))
		return _excess_variance


	def _summarize_timepoint(self, sequence_map, func):
		out = {}
		for key, val in sequence_map.items():
			out[key] = func(val.groupby("cell_line_name").days.agg(lambda v: len(v.unique())).drop('pDNA').values)
		return out


	def _get_median_guide_counts(self, unified_guide_map):
		return unified_guide_map.groupby("labels_inner").labels_outer.nunique().median()


	def _initialize_graph(self, max_learning_rate, dtype):
		self.printer.print('initializing graph')
		self.sess = tf.compat.v1.Session()
		self._learning_rate = tf.compat.v1.placeholder(shape=tuple(), dtype=dtype)
		self.run_dict = {
			self._learning_rate: max_learning_rate, 
			self._gene_effect_hierarchical: self._private_gene_effect_hierarchical
		}
		self.max_learning_rate = max_learning_rate
		self.persistent_handles = set([])


	def _get_gene_effect_mask(self, readcounts, sequence_map, guide_gene_map, dtype):
		# excludes genes in a cell line with reads from only one library
		self.printer.print('\nbuilding gene effect mask')

		masks = {
			key: readcounts[key]\
					.notnull()\
					.groupby(sequence_map[key].set_index("sequence_ID").cell_line_name)\
					.any()
					.groupby(guide_gene_map[key].set_index("sgrna").gene, axis=1)\
					.any()\
					.reindex(index=self.all_cells, columns=self.all_genes)\
					.fillna(False)
			for key in self.keys
		}

		combined_mask = None
		for mask in masks.values():
			if combined_mask is None:
				combined_mask = mask
			else:
				combined_mask |= mask.values
		missing_lines = combined_mask.any(axis=1).loc[lambda x: ~x].index
		if len(missing_lines):
			raise ValueError("no non-null reads found for %i cell lines in any library. Examples:\n%r" % (
				len(missing_lines), missing_lines[:5])
			)
		missing_genes = combined_mask.any().loc[lambda x: ~x].index
		if len(missing_genes):
			raise ValueError("no non-null reads found for %i genes in any library. Examples:\n%r" % (
				len(missing_genes), missing_genes[:5])
			)
		mask_count = combined_mask.sum().sum()
		combined_mask = combined_mask.astype(self.np_dtype)
		_gene_effect_mask = tf.constant(combined_mask.values, dtype=dtype, name="GE_mask")
		return _gene_effect_mask, mask_count



	def _get_days(self, sequence_map, dtype):   
		self.printer.print('\nbuilding doubling vectors')
		with tf.compat.v1.name_scope("days"):
			_days = {key: 
				tf.constant(
						  Chronos.default_timepoint_scale 
						* val.set_index('sequence_ID').loc[self.index_map[key]].days.astype(self.np_dtype).values, 
					dtype=dtype, 
					shape=(len(self.index_map[key]), 1), 
					name=key
				)
			for key, val in sequence_map.items()}
		for key in self.keys:
			self.printer.print("made days vector of shape %r for %s" %(
				_days[key].get_shape().as_list(), key))
		return _days


	def _get_late_tf_timepoints(self, readcounts, dtype):
		self.printer.print("\nbuilding late observed timepoints")
		_normalized_readcounts = {}
		_mask = {}
		for key in self.keys:
			normalized_readcounts_np = readcounts[key].loc[self.index_map[key], self.column_map[key]].copy()
			normalized_readcounts_np += 1e-10
			mask = pd.notnull(normalized_readcounts_np)
			_mask[key] = tf.constant(mask, dtype=tf.bool, name='NaN_mask_%s' % key)
			normalized_readcounts_np[~mask] = 0
			_normalized_readcounts[key] = self.get_persistent_input(dtype, normalized_readcounts_np, name='normalized_readcounts_%s' % key)
			self.printer.print("\tbuilt normalized timepoints for %s with shape %r (replicates X guides)" %(
				key, normalized_readcounts_np.shape))
		return _normalized_readcounts, _mask


	def _get_tf_measured_t0(self, readcounts, sequence_map, dtype):
		self.printer.print('\nbuilding t0 reads')
		_measured_t0 = {}
		_pdna_scale = {}
		with tf.compat.v1.name_scope("measured_t0"):
			for key in self.keys:
				rc = readcounts[key]
				sm = sequence_map[key]
				sm = sm[sm.cell_line_name == 'pDNA']
				batch = rc.loc[sm.sequence_ID]
				if batch.empty:
					raise ValueError("No sequenced entities are labeled 'pDNA', or there are no readcounts for those that are")
				if batch.shape[0] > 1:
					batch = batch.groupby(sm.pDNA_batch.values).mean().astype(self.np_dtype)
				else:
					batch = pd.DataFrame({self.pDNA_unique[key][0]: batch.iloc[0]}).T.astype(self.np_dtype)
				batch = batch.loc[self.pDNA_unique[key], self.column_map[key]]
				if batch.isnull().sum().sum() != 0:
					print(batch)
					raise RuntimeError("NaN values encountered in batched pDNA")
				rc = rc.loc[self.index_map[key], self.column_map[key]]
				batchsum = batch.sum(axis=1).iloc[self.batch_map[key]['gather_ind_inner']]
				batchmed = batch.median(axis=1).iloc[self.batch_map[key]['gather_ind_inner']]

				pdna_scale = batchsum.values * rc.median(axis=1)/batchmed.values
				_pdna_scale[key] = tf.constant(pdna_scale.values.reshape((-1, 1)), dtype=dtype, name="%s_pDNA_scale" % key)
				t0_normed = batch.divide(batch.sum(axis=1), axis=0).values + 1e-8
				_measured_t0[key] = tf.constant(t0_normed, name='measured_t0_%s' % key, dtype=dtype)
		return _measured_t0, _pdna_scale


##################    C  R  E  A  T  E       V  A  R  I  A  B  L  E  S   #######################

	def _get_t0_tf_variables(self, _measured_t0, dtype):
		self.printer.print("\nbuilding t0 reads estimate")

		v_t0 = {}
		_t0_core = {}
		_t0 = {}
		_t0_offset = {}
		_grouped_t0_offset = {}
		_t0_mask_input = {}
		_masked_t0 = {}
		denom = None

		with tf.compat.v1.name_scope("inferred_t0"):
			for key in self.keys:
				t0_normed = self.sess.run(_measured_t0[key], self.run_dict)
				# Note shape: guides X 1
				# the offset is shared across pDNA batches, but not library batches
				v_t0[key] = tf.Variable(
					np.zeros((t0_normed.shape[1], 1), dtype=self.np_dtype), 
					dtype=dtype, name='base_%s' % key
				)

				_t0_mask_input[key] = tf.compat.v1.placeholder(shape=(t0_normed.shape[1], 1), dtype=tf.bool,
					name="random_mask_%s" % key)
				_masked_t0[key] = tf.where(_t0_mask_input[key], v_t0[key], self.zero, name="masked_t0_%s" % key)
				# force the global mean of _t0_offset to be 0
				_t0_offset[key] = tf.subtract(_masked_t0[key], tf.reduce_mean(input_tensor=_masked_t0[key]), name="t0_offset_%s" % key)


				# holds per-gene sum of all pDNA offsets for the library
				# _grouped_t0_offset[key][gene] = SUM_<guide targets gene> (_t0_offset[key][guide])
				_grouped_t0_offset[key] = tf.transpose(a=tf.math.unsorted_segment_sum(
						_t0_offset[key],
						self.guide_map[key]['gather_ind_inner'],
						num_segments=self.ngenes,
						name='grouped_diff_%s' % key
				))

				counts = pd.Series(self.guide_map[key]['gather_ind_inner'])\
						.value_counts()\
						.reindex(range(self.ngenes))\
						.fillna(0)\
						.values\
						.reshape((1, -1))\
						.astype(self.np_dtype)
				if denom is None:
					denom = counts
				else:
					denom += counts

			_grouped_offset_denom = tf.constant(denom, dtype=dtype, name="denominator_for_mean_over_libraries")

			# holds the mean of offsets for all guides targeting the gene across all libraries
			# we force this to be effectively 0 for pDNA offsets we finally use
			_combined_t0_offset = tf.add_n([v for v in _grouped_t0_offset.values()], name="numerator_for_mean_over_libraries") \
										/ _grouped_offset_denom

			for key in self.keys:
				# Note: broadcast the t0_offset from guides X 1 to batch X guides
				_t0_core[key] = tf.multiply(
					self._measured_t0[key], 
					tf.exp(
						tf.transpose(a=_t0_offset[key])
						# for every guide, subtract the mean offset for all guides targetiing the same gene
					  - tf.gather(
							_combined_t0_offset, 
							self.guide_map[key]['gather_ind_inner'], 
							axis=1
						)
					),
					name="t0_core_%s" % key
				)
				
				_t0[key] = tf.gather(
					_t0_core[key] / tf.reshape(tf.reduce_sum(input_tensor=_t0_core[key], axis=1), shape=(-1, 1)),
						self.batch_map[key]['gather_ind_inner'], 
						axis=0, 
						name='t0_read_est_%s' % key
						)


				self.printer.print("made t0 batch with shape %r for %s" %(
					t0_normed.shape, key))

		return v_t0, _t0_core, _t0, _t0_offset, \
				_grouped_t0_offset, _grouped_offset_denom, _combined_t0_offset, \
				_masked_t0, _t0_mask_input


	def _get_tf_guide_efficacy(self, dtype):        
		self.printer.print("building guide efficacy")
		with tf.compat.v1.name_scope("guide_efficacy"):
			v_guide_efficacy = tf.Variable(
				#last guide is dummy
				np.random.normal(size=(1, self.nguides+1), scale=.001).astype(self.np_dtype),
				name='base', dtype=dtype)
			_guide_efficacy_mask_input = tf.compat.v1.placeholder(shape=(1, self.nguides+1), dtype=tf.bool, 
				name="mask_input")
			_guide_efficacy_masked = tf.where(_guide_efficacy_mask_input, v_guide_efficacy, self.zero,
				name="masked")
			_guide_efficacy = tf.exp(-tf.abs(_guide_efficacy_masked), name='guide_efficacy')
			tf.compat.v1.summary.histogram("guide_efficacy", _guide_efficacy)
			self.printer.print("built guide efficacy: shape %r" %_guide_efficacy.get_shape().as_list())
		return v_guide_efficacy, _guide_efficacy_mask_input, _guide_efficacy


	def _get_tf_growth_rate(self, dtype):
		self.printer.print("building growth rate")
		with tf.compat.v1.name_scope("growth_rate"):
			v_growth_rate = { key: tf.Variable(
					np.random.normal(size=(self.nlines, 1), scale=.01, loc=1).astype(self.np_dtype),
					name='base_%s' % key, dtype=dtype
			) for key in self.keys}
			_line_presence_mask = {key: tf.constant( 
				np.array([s in self.cells[key] for s in self.all_cells], dtype=self.np_dtype).reshape((-1, 1)),
				name="line_presence_mask_%s" % key 
				) for key in self.keys}
			_line_presence_boolean = {key: tf.constant( np.array([s in self.cells[key] for s in self.all_cells], dtype=bool), 
														dtype=tf.bool,
														name="line_presence_boolean_%s" % key)
									for key in self.keys}
			_growth_rate_square = {key: (val * _line_presence_mask[key]) ** 2 for key, val in v_growth_rate.items()}
			_growth_rate = {key: tf.divide(val, tf.reduce_mean(input_tensor=tf.boolean_mask(tensor=val, mask=_line_presence_boolean[key])), 
									name="growth_rate_%s" % key)
								for key, val in _growth_rate_square.items()}
		self.printer.print("built growth rate: shape %r" % {key: val.get_shape().as_list() 
			for key, val in _growth_rate.items()})
		return v_growth_rate, _growth_rate, _line_presence_boolean


	def _get_tf_cell_efficacy(self, dtype):
		self.printer.print("\nbuilding cell line efficacy")
		with tf.compat.v1.name_scope("cell_efficacy"):
			v_cell_efficacy = { key: tf.Variable(
					np.random.normal(size=(self.nlines, 1), scale=.01, loc=0).astype(self.np_dtype),
									name='base_%s' % key, dtype=dtype)
					for key in self.keys}
			_cell_efficacy = {key: tf.exp(-tf.abs(v_cell_efficacy[key]),
							  name='%s' % key)
					for key in self.keys}
		self.printer.print("built cell line efficacy: shapes %r" % {key: v.get_shape().as_list() for key, v in _cell_efficacy.items()})
		return v_cell_efficacy, _cell_efficacy


	def _get_tf_screen_delay(self, initial_screen_delay, dtype):
		self.printer.print("building screen delay")
		with tf.compat.v1.name_scope("screen_delay"):
			v_screen_delay = tf.Variable(
							np.sqrt(Chronos.default_timepoint_scale * initial_screen_delay) * np.ones((1, self.ngenes), dtype=self.np_dtype),
							dtype=dtype, name="base")
			_screen_delay = tf.square(v_screen_delay)
		tf.compat.v1.summary.histogram("screen_delay", _screen_delay)
		self.printer.print("built screen delay")
		return v_screen_delay, _screen_delay


	def _get_tf_gene_effect(self, dtype):
		self.printer.print("building gene effect")

		with tf.compat.v1.name_scope("GE"):
			gene_effect_est = np.random.uniform(-.0001, .0001, size=(self.nlines, self.ngenes)).astype(self.np_dtype)
			gene_effect_est = gene_effect_est - gene_effect_est.mean(axis=0).reshape((1, -1))
			v_mean_effect = tf.Variable(
				np.random.uniform(-.0001, .0001, size=(1, self.ngenes)), 
				name='mean', dtype=dtype
			)
			with tf.compat.v1.name_scope("residue"):
				v_residue = tf.Variable(gene_effect_est, dtype=dtype, name='base') 
				_residue = tf.multiply(v_residue, self._gene_effect_mask, name="masked_base")
				if self._pretrained:
					_true_residue = _residue
				else:
					_true_residue =  tf.subtract(
						_residue,
						(tf.reduce_mean(input_tensor=_residue, axis=0, name="sum_over_guides"))[tf.newaxis, :],
						name="mean_centered"
						)
				  
			_combined_gene_effect = tf.add(v_mean_effect, _true_residue, name="GE")

			with tf.compat.v1.name_scope("library_effect"):
				v_library_effect = {
					key: tf.Variable(
						np.zeros((1, self.ngenes)).astype(self.np_dtype),
						name="%s" % key, dtype=dtype
						)
					for key in self.keys}


				gene_overlap_indicator = np.array([s in self.intersecting_genes for s in self.all_genes], 
					dtype=self.np_dtype).reshape((1, -1))
				_gene_overlap_indicated = tf.constant(gene_overlap_indicator, dtype=dtype, name="gene_overlap_indicator")
				library_mean_guides = {
					key: self.guide_gene_map[key]\
							.query("gene in %r" % list(self.intersecting_genes))
							.groupby("gene")\
							.sgrna\
							.count()\
							.mean()
					for key in self.keys
				}
				library_mean_guides = {
					key: val / sum(library_mean_guides.values())
					for key, val in library_mean_guides.items()
				}
				_library_effect_indicated = {key: v * _gene_overlap_indicated for key, v in v_library_effect.items()}
				_library_effect_mean = tf.add_n([library_mean_guides[key] * _library_effect_indicated[key]
									 for key in self.keys])
				_library_effect = {key: v - _library_effect_mean for key, v in _library_effect_indicated.items()}

			tf.compat.v1.summary.histogram("mean_gene_effect", v_mean_effect)

		self.printer.print("built core gene effect: %i cell lines by %i genes" %tuple(_combined_gene_effect.get_shape().as_list()))


		return v_mean_effect, v_residue, _residue, _true_residue, _combined_gene_effect, \
				v_library_effect, _library_effect


#############################    C  O  R  E      M  O  D  E  L    ##############################

	def _get_effect_days(self, _screen_delay, _days):
		self.printer.print("\nbuilding effective days")
		with tf.compat.v1.name_scope("effective_days"):
			_effective_days = {key: 
				tf.clip_by_value(val - _screen_delay, 0, 100, name=key)
			for key, val in _days.items()}

		self.printer.print("built effective days, shapes %r" % {key: val.get_shape().as_list() for key, val in _effective_days.items()})
		return _effective_days


	def _get_gene_effect_growth(self, _combined_gene_effect, _growth_rate, _library_effect):
		self.printer.print('\nbuilding gene effect growth graph nodes')
		with tf.compat.v1.name_scope('GE_growth'):

			_gene_effect_growth = {key: 
				tf.multiply(
					tf.add(_combined_gene_effect, _library_effect[key], name="GE_combined_%s" % key), 
					_growth_rate[key],
					name=key) 
			for key in self.keys}

		self.printer.print("built gene effect growth graph nodes, shapes %r" % {key: val.get_shape().as_list() 
			for key, val in _gene_effect_growth.items()})
		return _gene_effect_growth


	def _get_combined_efficacy(self, _cell_efficacy, _guide_efficacy):
		self.printer.print('\nbuilding combined efficacy')
		with tf.compat.v1.name_scope('efficacy'):
			_efficacy = {key: 
					tf.matmul(_cell_efficacy[key], tf.gather(_guide_efficacy, self.guide_map[key]['gather_ind_outer'], axis=1, name='guide_%s' % key),
				 name="combined_%s" % key)
				 for key in self.keys} #cell line by all guide matrix
			_selected_efficacies = {
				key: tf.gather(#expand to replicates in given library
						_efficacy[key], 
						self.replicate_map[key]['gather_ind_inner'],
						name="replicate_level_%s" % key
						)
				for key in self.keys
			}
		self.printer.print("built combined efficacy, shape %r" % {key: v.get_shape().as_list()for key, v in _efficacy.items()})
		self.printer.print("built expanded combined efficacy, shapes %r" % {key: val.get_shape().as_list() for key, val in _selected_efficacies.items()})
		return _efficacy, _selected_efficacies


	def _get_growth_and_fold_change(self, _gene_effect_growth, _effective_days, _selected_efficacies):
		self.printer.print("\nbuilding growth estimates of edited cells and overall estimates of fold change in guide abundance")
		_change = {}
		_growth = {}
		with tf.compat.v1.name_scope("FC"):
			for key in self.keys:

				_growth[key] = tf.gather( 
					tf.exp( 
						tf.gather(
							_gene_effect_growth[key], 
							self.replicate_map[key]['gather_ind_inner'],
							axis=0
						) * _effective_days[key]
					)-1,
					self.guide_map[key]['gather_ind_inner'],
					axis=1,
					name="growth_%s" %key
					)

				_change[key] = tf.add(
					np.float64(1.0), 
					tf.multiply(
						_selected_efficacies[key], 
						_growth[key],
						name="eff_mult"
					),
					name="FC_%s" % key  
				)
		self.printer.print("built growth and change")
		return _growth, _change


	def _get_abundance_estimates(self, _t0, _change):
		self.printer.print("\nbuilding unnormalized estimates of final abundance")
		_predicted_readcounts_unscaled = {key: tf.multiply(_t0[key], _change[key], name="out_%s" % key)
				 for key in self.keys}
		
		self.printer.print("built unnormalized abundance")

		self.printer.print("\nbuilding normalized estimates of final abundance")
		with tf.compat.v1.name_scope('out_norm'):
			_predicted_readcounts = {key: 
					self._pdna_scale[key]\
					* tf.divide((val + 1e-32), tf.reshape(tf.reduce_sum(input_tensor=val, axis=1), shape=(-1, 1)),
						name=key
						)
							for key, val in _predicted_readcounts_unscaled.items()}
		self.printer.print("built normalized abundance")
		return _predicted_readcounts_unscaled, _predicted_readcounts


#####################################    C  O  S  T    #########################################


	def _get_guide_regularization_alt(self, _guide_efficacy, dtype):
		self.printer.print('\nassembling guide efficacy regularization')
		with tf.compat.v1.name_scope("guide_efficacy_reg"):
			_guide_reg_cost = tf.reduce_mean(
						input_tensor= 1 - _guide_efficacy,
						name="guide_reg_cost"
			)
		return _guide_reg_cost


	def _get_library_batch_reg(self, _gene_effect, library_reg, dtype):
		group_indicators = {}
		with tf.compat.v1.name_scope("library_batch_reg"):
			for key, group in self.cells.items():
				group_indicator = pd.Series(np.ones(len(group)),
									index=group
					).reindex(self.all_cells).fillna(0).astype(self.np_dtype)

				group_indicators[key] = group_indicator

			_group_indicators = {key: tf.constant(val.values.reshape((-1, 1)), dtype, name="group_indicator_%s" % key)
									for key, val in group_indicators.items()}
			_indicator_product = {key: tf.multiply(val, _gene_effect, name="indicator_product_%s" % key) for key, val in _group_indicators.items()}

			_library_means = {key: tf.reduce_sum(val, axis=0, name="library_effect_sum")[tf.newaxis, :] / group_indicators[key].sum()
								for key, val in _indicator_product.items()}

			_library_reg = tf.add_n([
				library_reg * tf.reduce_mean(_library_means[key]**2, name="squared_means")
				for key in self.keys], name="library_reg")

		return _library_means, _library_reg



	def _get_smoothed_ge_regularization(self, v_mean_effect, _true_residue, kernel_width, dtype):
		self.printer.print("building smoothed regularization")
		kernel_size = int(6 * kernel_width)
		kernel_size = kernel_size + kernel_size % 2 + 1 #guarantees odd width
		kernel = np.exp( -( np.arange(kernel_size, dtype=self.np_dtype) - kernel_size//2 )**2/ (2*kernel_width**2) )
		kernel = kernel / kernel.sum()

		with tf.compat.v1.name_scope("smoothed_GE_reg"):
			_kernel = tf.constant(kernel, dtype=dtype, name='kernel')[:, tf.newaxis, tf.newaxis]
			_ge_argsort = tf.argsort(v_mean_effect[0], name="argsort")
			_residue_sorted = tf.gather(_true_residue, _ge_argsort, axis=1, name="sorted")[:, :, tf.newaxis]
			_residue_smoothed = tf.nn.convolution(input=_residue_sorted, filters=_kernel, padding='SAME', name="smoothed")
			_smoothed_presum = tf.square(_residue_smoothed, name="squared_smoothed")
		return _smoothed_presum


	def _get_t0_regularization(self, _t0_offset):
		self.printer.print("\nbuilding t0 reads regularization/cost")
		with tf.compat.v1.name_scope("t0_reg"):
			_t0_cost = {key:
				tf.reduce_mean( input_tensor=tf.square(_t0_offset[key]), 
					name='%s' %key)
				for key in self.keys
			}
		return _t0_cost


	def _get_nb2_cost(self, _excess_variance, _predicted_readcounts, _normalized_readcounts, _mask, dtype):
		self.printer.print('\nbuilding NB2 cost')
		
		with tf.compat.v1.name_scope('cost'):
			# the NB2 cost: (yi + 1/alpha) * ln(1 + alpha mu_i) - yi ln(alpha mu_i)
			# modified with constants and -mu_i - which makes it become the multinomial cost in the limit alpha -> 0
			_cost_presum = {key:
				(
										((_normalized_readcounts[key]+1e-6) + 1./_excess_variance[key]) * tf.math.log(
											1 + _excess_variance[key] * (_predicted_readcounts[key] + 1e-6)
									) -
										(_normalized_readcounts[key]+1e-6) * tf.math.log(
											(_excess_variance[key] * _predicted_readcounts[key] + 1e-6)
										)
									)
								for key in self.keys}

			readcounts = self.normalized_readcounts
			ev = self.excess_variance
			ev = {key: ev[key].loc[readcounts[key].index].values.reshape((-1, 1))
				for key in self.keys
			}
			_cost_constant = {key: tf.constant(np.nansum(
									
										((readcounts[key].values+1e-6) + 1./ev[key])\
										 * np.log(
											(1 + ev[key] * (readcounts[key].values + 1e-6))
									) -
										(readcounts[key].values+1e-6) * np.log(
											ev[key]*readcounts[key].values + 1e-6 
										)
									))
									 for key in self.keys}

			_scale = tf.compat.v1.placeholder(dtype=dtype, shape=(), name='scale')
			_cost =  _scale/len(self.keys) * (
				tf.add_n([tf.reduce_sum(input_tensor=tf.boolean_mask(tensor=v, mask=_mask[key]))
													 for key, v in _cost_presum.items()]
						)
				- tf.add_n(_cost_constant.values())
			)

			tf.compat.v1.summary.scalar("unregularized_cost", _cost)
			return _cost_presum, _cost_constant, _cost, _scale


	def _get_full_cost(self, dtype):
		self.printer.print("building other regularizations")
		with tf.compat.v1.name_scope('full_cost'):

			self._L1_penalty = self.gene_effect_L1 * tf.reduce_sum(
				input_tensor=tf.abs(
					tf.reduce_sum(
						input_tensor=self._combined_gene_effect*self._gene_effect_mask, 
						axis=1
					)
				) / self.mask_count,
				name="L1_penalty"
			) 

			self._L2_penalty = self.gene_effect_L2 * tf.reduce_sum(input_tensor=tf.square(self._combined_gene_effect),
				name="L2_penalty")/self.mask_count

			self._hier_penalty = self._gene_effect_hierarchical * tf.reduce_sum(input_tensor=tf.square(self._true_residue),
				name="hier_penalty")/self.mask_count

			self._growth_reg_cost = -self.growth_rate_reg * 1.0/len(self.keys) * tf.add_n([
							tf.reduce_mean( input_tensor=tf.math.log(tf.boolean_mask(tensor=v, mask=self._line_presence_boolean[key])) )
							for key, v in self._growth_rate.items()
																					], name="growth_reg_cost")

			self._guide_efficacy_reg = tf.compat.v1.placeholder(dtype, shape=())
			self.run_dict[self._guide_efficacy_reg] = self.guide_efficacy_reg

			self._guide_reg_cost = self._guide_efficacy_reg * self._total_guide_reg_cost
			self._smoothed_cost = self.gene_effect_smoothing * tf.reduce_mean(input_tensor=self._smoothed_presum)

			self._offset_reg = tf.compat.v1.placeholder(dtype, shape=())
			self.run_dict[self._offset_reg] = self.offset_reg
			self._update_run_dict_guide_masks(mask=False)
			self._t0_cost_sum = self._offset_reg * 1.0/len(self.keys) * tf.add_n(list(self._t0_cost.values()))


			_full_cost = self._cost + \
							self._L1_penalty + self._L2_penalty + \
							self._hier_penalty  + \
							self._guide_reg_cost + \
							self._growth_reg_cost + self._t0_cost_sum +\
							self._smoothed_cost + self._library_batch_cost

			tf.compat.v1.summary.scalar("L1_penalty", self._L1_penalty)
			tf.compat.v1.summary.scalar("L2_penalty", self._L2_penalty)
			tf.compat.v1.summary.scalar("hierarchical_penalty", self._hier_penalty)
		return _full_cost


#########################    F  I  N  A  L  I  Z  I  N  G    ###################################

	def smart_initialize(self, readcounts, sequence_map, guide_gene_map, cell_efficacy_guide_quantile, negative_control_sgrnas,
		screen_delay):
		cell_eff_est = {}
		gene_effect_est = {}
		for key in self.keys:
			self.printer.print('\t', key)
			sm = sequence_map[key]
			last_reps = extract_last_reps(sm)
			fc = calculate_fold_change(readcounts[key], sm, rpm_normalize=False)
			cell_eff_est[key] = self.cell_efficacy_estimate(fc, sm, last_reps, cell_efficacy_guide_quantile)
			gene_effect_est[key] = self.gene_effect_estimate(fc, sm,  guide_gene_map[key], last_reps, screen_delay)

		self.cell_efficacy = cell_eff_est

		gene_effect_numerator = pd.DataFrame(0, index=self.all_cells, columns=self.all_genes)
		gene_effect_denominator = pd.DataFrame(0, index=self.all_cells, columns=self.all_genes)
		for key, val in gene_effect_est.items():
			val = val.reindex(index=self.all_cells, columns=self.all_genes)
			gene_effect_numerator += val.fillna(0).values
			gene_effect_denominator += val.notnull().values
		self.gene_effect = gene_effect_numerator / gene_effect_denominator

	def gene_effect_estimate(self, fold_change, sequence_map, guide_gene_map, last_reps, screen_delay):
		'''
		Initial estimate of gene effect using the mean fold changes guides for each gene in the latest provided time points, per library
		'''
		mean_fold_change = fold_change\
								.loc[last_reps]\
								.groupby(sequence_map.set_index("sequence_ID").cell_line_name)\
								.mean()\
								.groupby(guide_gene_map.set_index("sgrna").gene, axis=1)\
								.mean()
		mean_fold_change.replace(0, 1e-3, inplace=True)
		denom = sequence_map.set_index("sequence_ID").loc[last_reps].groupby("cell_line_name").days.max() - screen_delay
		denom = denom.loc[mean_fold_change.index] * Chronos.default_timepoint_scale
		if (denom <= 0).any():
			raise ValueError("Some lines have no replicates with `days` post infection greater than `initial_screen_delay`")
		return pd.DataFrame(
			np.log(mean_fold_change.values) / denom.values.reshape((-1, 1)),
			index=mean_fold_change.index, columns=mean_fold_change.columns
		)


	def cell_efficacy_estimate(self, fold_change, sequence_map, last_reps, cell_efficacy_guide_quantile=.01):
		'''
		Estimate the maximum depletion possible in cell lines as the lowest X percentile guide fold-change in
		the last timepoint measured. Multiple replicates for a cell line at the same last timepoint are median-collapsed
		before the percentile is measured.
		'''
		fc = fold_change.loc[last_reps].groupby(sequence_map.set_index('sequence_ID').cell_line_name).median()
		medians = fc.median()
		#this breaks when medians.min() == medians.quantile(q); e.g. q% of guides are at 0 median fc
		#depleting_guides = medians.loc[lambda x: x < medians.quantile(cell_efficacy_guide_quantile)].index
		depleting_guides = medians.loc[lambda x: x <= medians.quantile(cell_efficacy_guide_quantile)].index
		#####
		# if medians.min() == medians.quantile(cell_efficacy_guide_quantile):
		# 	depleting_guides = medians.loc[lambda x: x <= medians.quantile(cell_efficacy_guide_quantile)].index # keep
		# else:
		# 	depleting_guides = medians.loc[lambda x: x < medians.quantile(cell_efficacy_guide_quantile)].index   
		#####
		cell_efficacy = 1 - fc[depleting_guides].median(axis=1)
		if (cell_efficacy <=0 ).any() or (cell_efficacy > 1).any() or cell_efficacy.isnull().any():
			raise RuntimeError("estimated efficacy outside bounds. \n%r\n%r" % (cell_efficacy.sort_values(), fc))

		return cell_efficacy


	def nan_check(self):
		#labeled data
		self.printer.print('verifying user inputs')
		for key in self.keys:
			if pd.isnull(self.sess.run(self._days[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._days[%s]" %key
			if pd.isnull(self.sess.run(self._t0[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._t0[%s]" %key
			if pd.isnull(self.sess.run(self._normalized_readcounts[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._excess_variance[%s]" %key
			if (self.sess.run(self._excess_variance[key], self.run_dict) < 0).sum().sum() > 0:
				assert False, "negative values found in self._excess_variance[%s]" %key

		#variables
		self.printer.print('verifying variables')
		if pd.isnull(self.sess.run(self._combined_gene_effect, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self._combined_gene_effect"
		if pd.isnull(self.sess.run(self.v_guide_efficacy, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self.v_guide_efficacy"
		if pd.isnull(self.sess.run(self._guide_efficacy, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self._guide_efficacy"

		#calculated terms
		self.printer.print('verifying calculated terms')
		for key in self.keys:
			if pd.isnull(self.sess.run(self.v_cell_efficacy[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self.v_cell_efficacy[%r]" % key
			if pd.isnull(self.sess.run(self._efficacy[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._efficacy[%r]" % key
			self.printer.print('\t' + key + ' _gene_effect')
			if pd.isnull(self.sess.run(self._gene_effect_growth[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._gene_effect_growth[%s]" %key
			self.printer.print('\t' + key + ' _selected_efficacies')
			if pd.isnull(self.sess.run(self._selected_efficacies[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._selected_efficacies[%s]" %key
			self.printer.print('\t' + key + '_predicted_readcounts_unscaled')
			if pd.isnull(self.sess.run(self._predicted_readcounts_unscaled[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._predicted_readcounts_unscaled[%s]" %key
			if (self.sess.run(self._predicted_readcounts_unscaled[key], self.run_dict) < 0).sum().sum() > 0:
				assert False, "negatives found in self._predicted_readcounts_unscaled[%s]" %key
			self.printer.print('\t' + key + ' _predicted_readcounts')
			df = self.sess.run(self._predicted_readcounts[key], self.run_dict)
			if np.sum(pd.isnull(df).sum()) > 0:
				assert False, "%i out of %i possible nulls found in self._predicted_readcounts[%s]" % (
					np.sum(pd.isnull(df).sum()), np.prod(df.shape), key
					)
			if np.sum((df < 0).sum()) > 0:
				assert False, "negative values found in predicted_readcounts[%s]" % key
			self.printer.print('\t' + key + ' _normalized_readcounts')
			if np.sum(pd.isnull(self.sess.run(self._normalized_readcounts[key], self.run_dict)).sum()) > 0:
				assert False, "nulls found in self._normalized_readcounts[%s]" %key
			min_normalized_readcounts = self.sess.run(self._normalized_readcounts[key], self.run_dict).min().min()
			if min_normalized_readcounts < 0:
				raise ValueError("Negative Reads Per Million (normalized_readcounts) found (%f)" % min_normalized_readcounts)

			min_predicted_readcounts = self.sess.run(self._predicted_readcounts[key], self.run_dict).min().min()
			if min_predicted_readcounts < 0:
				raise ValueError("Negative predicted normalized reads (predicted_readcounts) found (%f)" % min_predicted_readcounts)
			self.printer.print('\t' + key + ' _cost_presum')
			df = self.cost_presum[key]
			self.printer.print("sess run")
			if np.sum(pd.isnull(df).sum()) > 0:
				print(df)
				print()
				print(self.sess.run(
					 tf.math.log(1 + self._excess_variance_expanded[key] * 1e6 * self._predicted_readcounts[key]), self.run_dict)
				)
				print()
				print(self.sess.run(
					(
							self._normalized_readcounts[key]+1e-6) 
						* tf.math.log(self._excess_variance_expanded[key] 
						* (self._normalized_readcounts[key] + 1e-6) 
					 ), 
						self.run_dict)
				)
				raise ValueError("%i nulls found in self._cost_presum[%s]" % (pd.isnull(df).sum().sum(), key))
			self.printer.print('\t' + key + ' _cost')
			if pd.isnull(self.sess.run(self._cost, self.run_dict)):
				assert False, "Cost is null"
			self.printer.print('\t' + key + ' _full_costs')
			if pd.isnull(self.sess.run(self._full_cost, self.run_dict)):
				assert False, "Full cost is null"



	################################################################################################
	####################   T R A I N I N G    M  E  T  H  O  D  S    ###############################
	################################################################################################


	def _select_guides_to_drop(self):
		'''
		get a dict per linbrary containing an array of guide names to drop equalling 50% of guides
		for each gene (rounded down)
		'''
		return {
			key: np.concatenate(self.guide_gene_map[key].groupby("gene").sgrna.apply(lambda x: 
									  np.random.choice(x, size=int(np.ceil(len(x)/3)), replace=False)
									 ).values)
			for key in self.keys
		}

	def _get_guide_indices(self, guides):
		'''
		return the indices of the string guide names in the array `guides` within `self.all_guides`
		'''
		guides = np.array(guides)
		guides.sort()
		return np.searchsorted(self.all_guides, guides)


	def _get_guide_indices_in_library(self, guides, library):
		'''
		return an array of the tensor indices of the guide name strings in `guides` in `library`
		'''
		indices = self._get_guide_indices(guides)
		return self.guide_map[library]['reverse_map_outer'][indices]


	def _assemble_guide_masks(self, mask=True):
		mask_all = np.ones((1, len(self.all_guides)+1), dtype=bool)
		#note: for efficiency, library_masks are transposed (first dimension is guide)
		library_masks = {
			key: np.ones((len(self.guide_map[key]['gather_ind_outer']), 1), dtype=bool)
			for key in self.keys
		}
		if not mask:
			#unmask the guide parameters
			return library_masks, mask_all

		guides_to_drop = self._select_guides_to_drop()
		all_guides_to_drop = sorted(set.union(*[set(v) for v in guides_to_drop.values()]))
		indices = self._get_guide_indices(all_guides_to_drop)
		mask_all[0, indices] = 0
		for key in self.keys:
			indices = self._get_guide_indices_in_library(guides_to_drop[key], key)
			library_masks[key][indices, 0] = 0
		return library_masks, mask_all

	def _update_run_dict_guide_masks(self, mask=True):
		library_masks, mask_all = self._assemble_guide_masks(mask)
		self.run_dict[self._guide_efficacy_mask_input] = mask_all
		for key in self.keys:
			self.run_dict[self._t0_mask_input[key]] = library_masks[key]
		

	def step(self, ge_only=False):
		'''
		Train the model for one step. If `ge_only`, update only the gene effect estimate.
		'''
		if self._pretrained and self._is_model_loaded:
			self.sess.run(self._loaded_model_step, self.run_dict) 
		else:
			self._update_run_dict_guide_masks(mask=True)
			if ge_only:
				self.sess.run(self._ge_only_step, self.run_dict)
			else:
				self.sess.run(self._step, self.run_dict)
			self._update_run_dict_guide_masks(mask=False)
		self.epoch += 1


	def train(self, nepochs=301, starting_learn_rate=1e-4, burn_in_period=50, ge_only=0, report_freq=50,
			essential_genes=None, nonessential_genes=None, additional_metrics={}):
		'''
		Train the model for a fixed number of epochs.
		Parameters:
			nepochs (`int`): number of epochs to train
			`starting_learn_rate` (`float`): the learning rate to use at the beginning of training. Using a lower rate
				improves stability. The learning rate used will increase exponentially until it reaches the 
				model's max learning rate at `burn_in_period` epochs. Note that this is determined from the total  number
				of epochs the model has been trained for, not the number trained in this particular invocation of the 
				`train` method. These will differ if `train` is called more than once for the model.
			`burn_in_period` (`int`): epochs over which to increase the learning rate
			`ge_only` (`int`): the number of epochs to train gene effect only before training all parameters. Like
				`burn_in_period`, this refers to the total number of epochs trained, not the number trained in
				the current invocation of `train`.
			`report_freq` (`int`): how often to write training summary statistics to stdout, in epochs.
			`essential_genes` (iterable of `None`): optional list of positive control genes. Adds some QC parameters
				to the training statistics but does not affect behavior.
			`nonessential_genes` (iterable of `None`): optional list of negative control genes. Adds some QC parameters
				to the training statistics but does not affect behavior.
			`additional_metrics` (`dict` of `func`): optional functions. Should accept the gene effect matrix as their
				only required argument and return a string.

		'''
		if self._pretrained != self._is_model_loaded:
			raise RuntimeError("Model is built to use a pretrained model, but no model was loaded. Please use model.load()")        
		
		rates = np.exp(np.linspace(np.log(starting_learn_rate), np.log(self.max_learning_rate), burn_in_period))
		start_time = time()
		start_epoch = self.epoch
		for i in range(start_epoch, start_epoch + nepochs):
			try:
				self.learning_rate = rates[self.epoch]
			except IndexError:
				self.learning_rate = self.max_learning_rate

			self.step(ge_only=self.epoch < ge_only)

			if not i%report_freq:
				delta = time() - start_time
				completed = i+1 - start_epoch
				to_go = nepochs - completed
				projected = delta * to_go/completed

				if completed > 1:
					self.printer.print('%i epochs trained, time taken %s, projected remaining %s' % 
						(i+1, timedelta(seconds=round(delta)), timedelta(seconds=round(projected)))
					)
				self.printer.print('NB2 cost', self.cost)
				self.printer.print("Full cost", self.full_cost)
				self.printer.print('relative_growth_rate')
				for key, val in self.growth_rate.items():
					self.printer.print('\t%s max %1.3f, min %1.5f' % (
						key, val[val!=0].max(), val[val!=0].min()))
				self.printer.print('mean guide efficacy', self.guide_efficacy.mean())
				self.printer.print('t0_offset SD: %r' % [(key, self.t0_offset[key].std()) for key in self.keys]) 
				self.printer.print()
				ge = self.gene_effect
				self.printer.print('gene mean', ge.mean().mean())
				self.printer.print('SD of gene means', ge.mean().std())
				self.printer.print("Mean of gene SDs", ge.std().mean())
				for key, val in additional_metrics.items():
					self.printer.print(key, val(ge))
				if essential_genes is not None:
					self.printer.print("Fraction Ess gene scores in bottom 15%%:", (ge.rank(axis=1, pct=True)[essential_genes] < .15).mean().mean()
					)
					self.printer.print("Fraction Ess gene medians in bottom 15%%:", (ge.median().rank(pct=True)[essential_genes] < .15).mean()
					)
				if nonessential_genes is not None:
					self.printer.print("Fraction Ness gene scores in top 85%%:", (ge.rank(axis=1, pct=True)[nonessential_genes] > .15).mean().mean()
					)
					self.printer.print("Fraction Ness gene medians in top 85%%:", (ge.median().rank(pct=True)[nonessential_genes] > .15).mean()
					)

				self.printer.print('\n\n')


#########################               I  /  O              ###################################           
				
				
	def load(self, gene_effect, guide_efficacy, t0_offset, screen_delay, cell_efficacy, 
		library_effect):
		'''

		'''
		if self._pretrained != True:
			raise RuntimeError("Model is built to train without existing data. \
To load a pretrained model, you must reinitialize Chronos with `pretrained=True`")
		
		# convert cell_line_efficacy output into dictionary of cell lines to sets of library names
		cells_to_libraries_screened = dict()
		for cell_id in cell_efficacy.index:
			# only keep the libraries that are present in the newest data (self.keys)
			cells_to_libraries_screened[cell_id] = set(
				cell_efficacy\
				.loc[cell_id, ~cell_efficacy.loc[cell_id, :].isna()]\
				.index
			).intersection(self.keys)
		
		missing = set(self.keys) - set.union(*[libs for libs in cells_to_libraries_screened.values()])
		if len(missing):
			raise ValueError(
				"Data contains libraries that are not present in the pretrained model: %r. Please load a pretrained model \
that includes all the libraries in the new screen(s), run Chronos without a pretrained model, or exclude those libraries from \
your data" % missing
			)
		
		mask = {}
		for cell in self.all_cells:
			if cell in cells_to_libraries_screened:
				libraries = set([key for key in self.keys if cell in self.cells[key]]).union(cells_to_libraries_screened[cell])
			else:
				libraries = [key for key in self.keys if cell in self.cells[key]]
			covered_genes = sorted(set.intersection(*[set(self.genes[key]) for key in libraries]))
			mask[cell] = pd.Series(1, index=covered_genes, dtype=self.np_dtype)  
		# mask must be constructed so cell lines not present in original data are not NAed
		mask = pd.DataFrame(mask)\
				.T\
				.reindex(index=self.all_cells)\
				.fillna(1)\
				.reindex(columns=self.all_genes)\
				.fillna(0)
		_gene_effect_mask = tf.constant(mask.values, dtype=self.np_dtype)
		mask_count = (mask == 1).sum().sum()
		self._gene_effect_mask = _gene_effect_mask
		self.mask_count = mask_count

		# want to use the gene_effect matrix and not v_mean_effect's mean because the training data's masking is relevant
		means = gene_effect.mean().reindex(index=self.all_genes).values.reshape(1, -1)
		self.sess.run(self.v_mean_effect.assign(means))
		
		self.guide_efficacy = guide_efficacy

		self.t0_offset = t0_offset
		
		self.screen_delay = screen_delay
		
		self.library_effect = library_effect
		
		self._is_model_loaded = True
		self.printer.print('Chronos model loaded')
		self.printer.print('ready to train')
	
	
	def import_model(self, directory):
		'''
		Quickly load a subset of Chronos parameters from a directory. 
		The directory must contain the files "gene_effect.hdf5",
		"guide_efficacy.csv", "cell_line_efficacy.csv", and "library_effect.csv". Optionally it can
		contain "screen_delay.csv"; otherwise, the delay will be assumed to be 3 days.
		'''
		assert os.path.isdir(directory), "Directory %r does not exist" % directory
		dir_files = os.listdir(directory)       
		for filename in [
			'guide_efficacy.csv', "t0_offset.csv",
			'cell_line_efficacy.csv', 'library_effect.csv'
		]:
			assert filename in dir_files,"Cannot locate file {} in target directory {}".format(filename, directory)
		try:
			gene_effect = read_hdf5(os.path.join(directory, 'gene_effect.hdf5'))
		except FileNotFoundError:
			if "gene_effect.csv" in os.listdir(directory):
				print("reading gene effect from CSV, this may take a minute")
				gene_effect = pd.read_csv(os.path.join(directory, 'gene_effect.csv'), index_col=0)
			else:
				raise FileNotFoundError("neither gene_effect.hdf5 nor gene_effect.csv are present in %r" % directory)
		missing = sorted(set(self.all_genes) - set(gene_effect.columns))
		if len(missing):
			raise IndexError("Not all genes found in the guide gene map are present in the file %s. Example missing genes:\n%r" %(
				os.path.join(directory, "gene_effect.hdf5"), missing[:5]+missing[-5:]))
		guide_efficacy = pd.read_csv(os.path.join(directory, 'guide_efficacy.csv'), 
			index_col=0)["efficacy"]
		t0_offset = pd.read_csv(os.path.join(directory, 't0_offset.csv'), 
			index_col=0)
		cell_line_efficacy = pd.read_csv(os.path.join(directory, 'cell_line_efficacy.csv'), index_col=0)
		if "screen_delay.csv" in dir_files:
			screen_delay = pd.read_csv(os.path.join(directory, 'screen_delay.csv'), 
				index_col=0)["screen_delay"]
		else:
			screen_delay = pd.Series(3, index=gene_effect.columns)
		library_effect = pd.read_csv(os.path.join(directory, 'library_effect.csv'), index_col=0)        
		self.load(gene_effect, guide_efficacy, t0_offset, screen_delay, cell_line_efficacy, library_effect)
		   
		

	def save(self, directory, overwrite=False, include_inputs=True, include_outputs=True):
		'''
		Writes all the necessary model parameters as files in a directory. 
		Parameters:
			directory (`str`): path to desired output. The directory will be created.
			overwrite (`bool`): if False, and `directory` already exists, an error will be raised.
			include_inputs (`bool`): whether to write the data inputs readcounts, guide_map, sequence_map, and 
				negative_control_sgrnas to the directory.
			include_outputs (`bool`): whether to write calculated terms to the directory. These are the estimated
				log fold-change and estimated readcounts. They are not necessary to restore the model but are
				used in QCing the model's output.
		Always writes:
			'gene_effect.hdf5': the matrix of inferred gene effects
			'guide_efficacy.csv': the estimated efficacy of gene KO of each sgRNA
			'cell_line_efficacy.csv': the estimated efficacy of gene KO in each cell line in each library, with one column per 
				library
			'cell_line_growth_rate.csv': the estimated relative growth of each cell line, formatted like cell_line_efficacy.csv.
			'screen_excess_variance.csv': the per-screen, per-library estimated overdispersion parameter of the NB2 model,
				same format.
			'screen_delay.csv': The onset time for the viability effect from gene KO, untrained by default.
			'library_effect.csv': the estimated bias for each gene in each library, for genes present in all libraries.
			't0_offset.csv': the estimated fractional difference in the relative abundance of each sgRNA in each library
				where present from that reported in the pDNA.
			'paramdeters.json': the hyperparameters passed to `__init__` when the model was created. Also includes the 
				calculated `cost` and `full_cost` for reference.
		If `include_inputs`:
			'<library>_readcounts.hdf5': for each library, the matrix of observed readcounts. If Chronos mnormalized 
				these, the written version will also be normalized.
			'<library>_guide_gene_map.csv': for each library, the guide to gene map.
			'<library>_sequence_map.csv': for each library, the sequence map. The pDNA batch labels will have the library
				names added to guarantee uniqueness.
		if `include_outputs`:
			'<library>_predicted_readcounts.hdf5': for each library, the matrix of Chronos' predicted readcounts.
			'<library>_predicted_lfc.hdf5': for each library, the matrix of predicted sgRNA log fold-change (from observed
				relative pDNA abundance). 
		'''
		if os.path.isdir(directory) and not overwrite:
			raise ValueError("Directory %r exists. To overwrite contents, pass `overwrite=True`" % directory)
		elif not os.path.isdir(directory):
			os.mkdir(directory)

		write_hdf5(self.gene_effect, os.path.join(directory, "gene_effect.hdf5"))
		pd.DataFrame({"efficacy": self.guide_efficacy}).to_csv(os.path.join(directory,  "guide_efficacy.csv"))
		pd.DataFrame(self.cell_efficacy).to_csv(os.path.join(directory,  "cell_line_efficacy.csv"))
		pd.DataFrame(self.growth_rate).to_csv(os.path.join(directory,  "cell_line_growth_rate.csv"))
		pd.DataFrame(self.excess_variance).to_csv(os.path.join(directory,  "screen_excess_variance.csv"))
		pd.DataFrame({'screen_delay': self.screen_delay}).to_csv(os.path.join(directory,  "screen_delay.csv"))
		self.library_effect.to_csv(os.path.join(directory,  "library_effect.csv"))
		pd.DataFrame(self.t0_offset).to_csv(os.path.join(directory,  "t0_offset.csv"))

		parameters = {
			"guide_efficacy_reg": self.guide_efficacy_reg,
			"gene_effect_L1": self.gene_effect_L1,
			"gene_effect_L2": self.gene_effect_L2,
			"gene_effect_hierarchical": self.gene_effect_hierarchical,
			"growth_rate_reg": self.growth_rate_reg,
			"offset_reg": self.offset_reg,
			"gene_effect_smoothing": self.gene_effect_smoothing,
			"kernel_width": self.kernel_width,
			"cell_efficacy_guide_quantile": self.cell_efficacy_guide_quantile,
			"library_batch_reg": self.library_batch_reg,
			"cost": self.cost,
			"full_cost": self.full_cost
		}
		with open(os.path.join(directory, "parameters.json"), "w") as f:
			f.write(json.dumps(parameters))

		if include_inputs:
			for key in self.keys:
				write_hdf5(self.readcounts[key], os.path.join(directory, "%s_readcounts.hdf5" % key))
				self.guide_gene_map[key].to_csv(os.path.join(directory, "%s_guide_gene_map.csv" % key), index=None)
				self.sequence_map[key].to_csv(os.path.join(directory, "%s_sequence_map.csv" % key), index=None)
				if key in self.negative_control_sgrnas:
					pd.DataFrame(self.negative_control_sgrnas[key]).to_csv(os.path.join(directory, "%s_negative_control_sgrnas.csv" % key),
						index=None)

		if include_outputs:
			predicted_readcounts = self.predicted_readcounts
			fc = self.estimated_fold_change
			for key in self.keys:
				write_hdf5(predicted_readcounts[key], os.path.join(directory, "%s_predicted_readcounts.hdf5" % key))
				write_hdf5(pd.DataFrame(np.log2(fc[key].values), index=fc[key].index, columns=fc[key].columns),
					os.path.join(directory, "%s_predicted_lfc.hdf5" % key)
					)

	def snapshot(self):
		'''
		record tensorflow summaries at the current epochj
		'''
		try:
			summary, cost = self.sess.run([self._merged, self._full_cost], self.run_dict)
			self.writer.add_summary(summary, self.epoch)
		except AttributeError:
			raise RuntimeError("missing writer for creating snapshot, probably because no log directory was supplied to Chronos")


	def __del__(self):
		for handle in self.persistent_handles:
			tf.compat.v1.delete_session_tensor(handle)
		self.sess.close()


	################################################################################################
	########################    E  V  A  L  U  A  T  I  O  N    ####################################
	################################################################################################

	def compute_tf_gradients(self, tf_cost, tf_variable):
		'''
		get the tensorflow gradients of a scalar `tf_cost` w.r.t. a tensor `tf_variable`
		This function accepts only tf nodes for `tf_cost` and `tf_variable`
		Returns:
			gradients (`tf.tensor`): a tensor with the same shape as `tf_variable`
		'''
		return self.sess.run(tf.gradients(tf_cost, tf_variable), self.run_dict)

	def compute_gradients(self, cost, variable):
		'''
		get the gradients of the scalar `cost` with respect to the pandas `variable`.
		This function accepts string names only
		Returns:
			gradients (`pd.Series` or `pd.DataFrame` or `dict` of same): an object with the same shape and indices as `variable`
		'''
		costs = {
			"cost": self._cost, 
			"total_cost": self._full_cost,
			"L1_penalty": self._L1_penalty,
			"L2_penalty": self._L2_penalty,
			"hierarchical_penalty": self._hier_penalty,
			"growth_reg_cost": self._growth_reg_cost,
			#"guide_efficacy_reg": self._guide_reg_cost,
			"smoothed_cost": self._smoothed_cost,
			"t0_cost": self._t0_cost_sum,
		}
		if not cost in costs:
			raise ValueError("`cost` must be a string and one of %r" % sorted(costs.keys()))
		variables = {
			"gene_effect": (self._combined_gene_effect, self.all_cells, self.all_genes),
			"gene_effect_deviation": (self._true_residue, self.all_cells, self.all_genes),
			"library_effect": (self.v_library_effect, self.all_genes, None),
			"mean_effect": (self.v_mean_effect, self.all_genes, None),
			"guide_efficacy": (self._guide_efficacy, self.all_guides, None),
			"growth_rate": (self._growth_rate, self.all_cells, None),
			"guide_t0_offset": (self._t0_offset, self.column_map, None)
		}
		if not variable in variables:
			raise ValueError("`variables` must be a string and one of %r" % sorted(variables.keys()))
		tf_cost = costs[cost]
		tf_variable, index, columns = variables[variable]
		if not isinstance(tf_variable, dict):
			array = np.squeeze(self.compute_tf_gradients(tf_cost, tf_variable))
			if variable == "guide_efficacy":
				array = array[:-1]
			if columns:
				return pd.DataFrame(array, index=index, columns=columns)
			else:
				return pd.Series(array, index=index)
		else:
			array = {key: np.squeeze(self.compute_tf_gradients(tf_cost,  tf_variable[key]))
						for key in self.keys}
			if isinstance(index, dict):		
				return {key: pd.Series(array[key], index=index[key])
					for key in self.keys}
			else:
				return pd.DataFrame(array, index=index)
	################################################################################################
	########################    A  T  T  R  I  B  U  T  E  S    ####################################
	################################################################################################

	def inverse_efficacy(x):
		'''
		converts a float `x` in (0, 1] to the value which should
		be assigned to the underlying tf variable to produce
		an equivalent efficacy
		'''
		if not all((x <= 1) & (x > 0)):
			raise ValueError("efficacies must be greater than 0 and less than or equal to 1, received %r" % x)
		return -np.log(x)

	def __repr__(self):
		return \
"Chronos model with libraries %r, %i total cell lines, %i total genes, trained for %i epochs" % (
			self.keys, self.nlines, self.ngenes, self.epoch
		)

	
	@property
	def cell_efficacy(self):
		'''
		A `dict` of `pandas.Series` indexed by cell line containing the efficacy
		of the screen of the line in each library (where present)
		'''
		out = {key: 
			pd.Series(self.sess.run(self._cell_efficacy[key])[:, 0], index=self.all_cells).loc[self.cells[key]] for key in self.keys
		}
		for v in out.values():
			v.index.name = "cell_line_name"
		return pd.DataFrame(out)
	@cell_efficacy.setter
	def cell_efficacy(self, desired_efficacy):
		for key in self.keys:
			missing = set(self.cells[key]) - set(desired_efficacy[key].index)
			if len(missing) > 0:
				raise ValueError("tried to assign cell efficacy for %s but missing %r" % (key, missing))
			try:
				self.sess.run(self.v_cell_efficacy[key].assign(
					Chronos.inverse_efficacy(desired_efficacy[key].reindex(self.all_cells).fillna(1).values).reshape((-1, 1))
				))
			except ValueError as e:
				print(key)
				print(desired_efficacy[key].sort_values())
				raise e

	
	@property
	def cost(self):
		'''
		The mean NB2 cost of the difference between the model's estimated readcounts and the observed
		readcounts.
		'''
		return self.sess.run(self._cost, self.run_dict)


	@property
	def cost_presum(self):
		'''
		The NB2 cost of the difference between the model's estimated readcounts and the observed
		readcounts for each value estimated. Returns a per-library `dict` of `pandas.DataFrames`
		indexed by sequence IDs (rows) and sgRNAs (columns)
		'''
		return {key: pd.DataFrame(self.sess.run(self._cost_presum[key], self.run_dict), 
							  index=self.index_map[key], columns=self.column_map[key])
				for key in self.keys}
	

	@property
	def days(self):
		'''
		`dict` of `pandas.Series` indexed by sequence_ID containing the days after infection
		the sequence was observed. Constant. 
		'''
		out = {key: pd.Series(self.sess.run(self._days)[:, 0], index=self.index_map[key])
				for key in self.keys}
		for v in out.values():
			v.index.name = "sequence_ID"


	@property
	def estimated_fold_change(self):
		'''
		The model's estimate of the fold change from MEASURED pDNA abundance
		'''
		output = self.predicted_readcounts_unscaled
		measured_t0 = self.measured_t0
		mapper = {
			key: pd.DataFrame(self.batch_map[key])\
				.set_index("labels_outer")\
				.loc[output[key].index, "labels_inner"]
			for key in self.keys
		}
		return {
			key: pd.DataFrame(
					output[key].values \
					/ measured_t0[key].loc[mapper[key]].values,
				index=output[key].index, 
				columns=output[key].columns
			)
			for key in self.keys
		}


	@property
	def full_cost(self):
		'''
		The total scalar quantity optimized by Chronos, including all regularizing penalties.
		'''
		return self.sess.run(self._full_cost, self.run_dict)


	@property
	def gene_effect(self):
		'''
		`pandas.DataFrame` indexed by cell line and gene. The relative change in growth
		rate induced by successful knock out of the gene in the given line.
		For most use cases this is the quantity of interest.
		'''
		mask = self.sess.run(self._gene_effect_mask)
		array = self.sess.run(self._combined_gene_effect)
		array[mask == 0] = np.nan
		out = pd.DataFrame(array,
					index=self.all_cells, columns=self.all_genes
							)
		out.index.name = "cell_line_name"
		out.columns.name = "gene"
		return out

	@gene_effect.setter
	def gene_effect(self, desired_effect):
		mask = self.sess.run(self._gene_effect_mask)
		de = desired_effect.reindex(index=self.all_cells, columns=self.all_genes)
		if ((desired_effect.notnull() + mask) == 1).any().any():
			warn(
			 "received some nonnull values for genes in cell lines that have no guides targeting them, or inappropriate null values"
			)
		de[mask == 0] = np.nan 
		means = de.mean().values.reshape((1, -1))
		residue = pd.DataFrame(de.values - means).fillna(0).values
		self.sess.run(self.v_mean_effect.assign(means))
		self.sess.run(self.v_residue.assign(residue))


	@property
	def gene_effect_hierarchical(self):
		return self.run_dict[self._gene_effect_hierarchical]

	@gene_effect_hierarchical.setter
	def gene_effect_hierarchical(self, desired_gene_effect_hierarchical):
		self.run_dict[self._gene_effect_hierarchical] = desired_gene_effect_hierarchical
		

	@property
	def guide_efficacy(self):
		'''
		A `pandas.Series` containing the estimated guide KO efficacy in (0, 1]
		for each guide.
		'''
		out = pd.Series(self.sess.run(self._guide_efficacy, self.run_dict)[0][:-1], index=self.all_guides)
		out.index.name = "sgrna"
		return out

	@guide_efficacy.setter
	def guide_efficacy(self, desired_efficacy):
		self.sess.run(self.v_guide_efficacy.assign(
			Chronos.inverse_efficacy(np.array(list(desired_efficacy.loc[self.all_guides]) + [1e-16])).reshape((1, -1)) + 1e-16
		))
	

	@property
	def growth_rate(self):
		'''
		A `dict` of `pandas.Series` containing the unperturbed cell growth rate relative to 
		the growth of each cell line in each library. 
		'''
		out = {key: pd.Series(self.sess.run(self._growth_rate[key])[:, 0], index=self.all_cells).loc[self.cells[key]]
				for key in self.keys}
		for v in out.values():
			v.index.name = "cell_line_name"
		return pd.DataFrame(out)

	@growth_rate.setter
	def growth_rate(self, desired_growth_rate):
		for key in desired_growth_rate:
			missing = set(self.cells[key]) - set(desired_growth_rate[key].dropna().index)
			if len(missing) > 0:
				raise ValueError("tried to assign cell efficacy for %s but missing %r" % (key, missing))
			assert (desired_growth_rate[key].dropna() > 0).all(), "growth rate must be > 0"
			self.sess.run(self.v_growth_rate[key].assign(
				np.sqrt(desired_growth_rate[key].reindex(self.all_cells).fillna(1)).values.reshape((-1, 1)))
			)
	

	@property
	def efficacy(self):
		'''
		A `dict` of `pandas.DataFrame` containing the outer product of `guide_efficacy` 
		and `cell_efficacy` 
		'''
		out = {key: pd.DataFrame(self.sess.run(self._efficacy[key], self.run_dict),
					index=self.all_cells, columns=self.all_guides)
				for key in self.keys}
		for v in out.values():
			v.index.name = "cell_line_name"
			v.columns.name = "gene"
		return out
	

	@property
	def excess_variance(self):
		'''
		`dict` of `pandas.Series` indexed by sequence_ID containing the NB2 overdispersion parameter alpha
		for each replicate in each library.
		'''
		out = {key: pd.Series(self.sess.run(self._excess_variance[key], self.run_dict)[:, 0],
			index=self.index_map[key]
			) for key in self.keys}
		for v in out.values():
			v.index.name = "sequence_ID"
		return pd.DataFrame(out)
	@excess_variance.setter
	def excess_variance(self, desired_excess_variance):
		assert set(desired_excess_variance.columns) == set(self.keys), "t0_offset must have an entry for all and only %r" % self.keys
		for key in self.keys:
			self.sess.run(
				self._excess_variance[key].assign(desired_excess_variance[key].loc[self.index_map[key]].values.reshape((-1,  1)))
			)		

	
	@property
	def learning_rate(self):
		'''
		The current learning rate (step size) for the optimizer
		'''
		return self.run_dict[self._learning_rate]
	@learning_rate.setter
	def learning_rate(self, desired_learning_rate):
		self.run_dict[self._learning_rate] = desired_learning_rate


	@property
	def library_means(self):
		'''
		`pandas.DataFrame` holding the deviation of each gene in each library from the oveall
		gene mean. 
		'''
		out = pd.DataFrame({key: v[0]
			for key, v in self.sess.run(self._library_means).items()
		}, index=self.all_genes)
		out.index.name = "gene"
		out.columns.name = "library"
		return out


	@property
	def library_effect(self):
		'''
		`pandas.DataFrame` of the gene effect per library, indexed by genes with one column per library
		'''
		out = pd.DataFrame({key: v[0]
			for key, v in self.sess.run(self._library_effect).items()
		}, index=self.all_genes)
		out.index.name = "gene"
		out.columns.name = "library"
		return out

	@library_effect.setter
	def library_effect(self, desired_effect):
		if isinstance(desired_effect, pd.DataFrame):
			if len(set(desired_effect.columns) - set(self.keys)) and not len(set(desired_effect.index) - set(self.keys)):
				desired_effect = desired_effect.T
		for key in self.keys:
			self.sess.run(
				self.v_library_effect[key].assign(desired_effect[key].reindex(self.all_genes).fillna(0).values.reshape((1, -1)))
			)


	@property
	def mask(self):
		'''
		`dict` of `pandas.DataFrame` indexed by sequence ID and guide. Each dataframe is a boolean
		representing whether the corresponding cell in the readcount matrix `normalized_readcounts`
		should be treated as null. Constant.
		'''
		out = {
			key: pd.DataFrame(
				self.sess.run(self._mask[key]),
				index=self.index_map[key],
				columns=self.column_map[key]
			)
			for key in self.keys
		}
		for v in out.values():
			v.index.name = "sequence_ID"
			v.columns.name = "sgrna"
		return out
	

	@property
	def measured_t0(self):
		'''
		`dict` of `pandas.DataFrame` indexed by pDNA batch and guide containing the normalized
		observed pDNA counts (each row sums to 1). 
		'''
		out = {key: pd.DataFrame(self.sess.run(self._measured_t0[key], self.run_dict),
			index=self.pDNA_unique[key], columns=self.column_map[key]
			) for key in self.keys}
		for v in out.values():
			v.index.name = "pDNA_batch"
			v.columns.name = "sgrna"
		return out


	@property
	def normalized_readcounts(self):
		'''
		`dict` of `pandas.DataFrame` indexed by pDNA batch and guide containing the readcounts.
		Readcounts are normalized during initialization according to whether `negative_control_sgrnas`
		are provided. 
		These are the target which Chronos chooses its parameters to match.
		Constant, but not actually a TF constant due to graph memory limits. Instead, for efficiency
		these are passed to placeholders using persistent inputs in `get_persistent_input`
		'''
		mask = self.mask
		out = {key: 
				pd.DataFrame(
					np.array(self.sess.run(v, self.run_dict)),
					index=self.index_map[key], 
					columns=self.column_map[key]
				).mask(~mask[key])
			for key, v in self._normalized_readcounts.items()}
		for v in out.values():
			v.index.name = "sequence_ID"
			v.columns.name = "sgrna"
		return out


	@property
	def predicted_readcounts_unscaled(self):
		'''
		`dict` of `pandas.DataFrame` indexed by pDNA batch and guide containing the raw predicted
		readcounts from Chronos. These must be rescaled to match the total abundance of `normalized_readcounts`.
		'''
		mask = self.mask
		nout = {key: 
				pd.DataFrame(
					np.array(self.sess.run(v, self.run_dict)),
					index=self.index_map[key], 
					columns=self.column_map[key]
				).mask(~mask[key])
			for key, v in self._predicted_readcounts_unscaled.items()}
		for v in nout.values():
			v.index.name = "sequence_ID"
			v.columns.name = "sgrna"
		return nout

	
	@property
	def predicted_readcounts(self):
		'''
		`dict` of `pandas.DataFrame` indexed by pDNA batch and guide containing the scaled predicted
		readcounts from Chronos. These are generated from `out` by setting the sum of each row equal to
		the `pDNA_scale` for that row. Note that this assumes the `normalized_readcounts` have already 
		been aligned with the pDNA, which is done during initialization. 
		'''
		mask = self.mask
		out = {key: 
				pd.DataFrame(
					np.array(self.sess.run(v, self.run_dict)),
					index=self.index_map[key], 
					columns=self.column_map[key]
				).mask(~mask[key])
			for key, v in self._predicted_readcounts.items()}
		for v in out.values():
			v.index.name = "sequence_ID"
			v.columns.name = "sgrna"
		return out
	
	
	@property
	def screen_delay(self):
		'''
		`pandas.Series` indexd by gene giving a per-gene delay between infection and onset of a viability phenotype. 
		Currently a constant for all genes, 3 days by default.
		'''
		out = pd.Series(self.sess.run(self._screen_delay/Chronos.default_timepoint_scale, self.run_dict)[0], index=self.all_genes)
		out.index.name = "gene"
		return out
	@screen_delay.setter
	def screen_delay(self, desired_screen_delay):
		assert (desired_screen_delay >= 0).all(), "Screen delay must be >= 0 with no null values"
		self.sess.run(
			self.v_screen_delay.assign(np.sqrt(Chronos.default_timepoint_scale * desired_screen_delay.loc[self.all_genes].values.reshape((1, -1))))
		)

	
	@property
	def t0(self):
		'''
		`dict` of `pandas.DataFrame` indexed by sequence ID and sgRNA giving Chronos' estimate of the initial abundance of
		the sgRNA, which may differ from the observed abundance given `measured_t0`. Expanded from `t0_core` after renormalizing.
		'''
		out = {key: pd.DataFrame(self.sess.run(self._t0[key], self.run_dict),
			index=self.index_map[key], columns=self.column_map[key]
			) for key in self.keys}
		for v in out.values():
			v.index.name = "sequence_ID"
			v.columns.name = "sgrna"
		return out
	

	@property
	def t0_core(self):
		'''
		`dict` of `pandas.DataFrame` indexed by pDNA batch and sgRNA giving Chronos' estimate of the initial abundance of
		the sgRNA, which may differ from the observed abundance given `measured_t0`.
		'''
		out = {key: pd.DataFrame(self.sess.run(self._t0_core[key], self.run_dict),
			index=self.pDNA_unique[key], columns=self.column_map[key]
			) for key in self.keys}
		for v in out.values():
			v.index.name = "pDNA_batch"
			v.columns.name = "sgrna"
		return out

	
	@property
	def t0_offset(self):
		'''
		`dict` of `pandas.Series` indexed by sgRNA giving Chronos' estimate of how different the real 
		initial abundance is from that measured in `measured_t0`.
		'''
		out = {key:
			pd.Series(self.sess.run(self._t0_offset[key], self.run_dict)[:, 0],
				index=self.column_map[key])
			for key in self.keys
			}
		for v in out.values():
			v.index.name = "sgrna"
		return out
	@t0_offset.setter
	def t0_offset(self, desired_offset):
		for key in self.keys:
			self.sess.run(self.v_t0[key].assign(desired_offset[key].loc[self.column_map[key]].values.reshape((-1, 1))))

