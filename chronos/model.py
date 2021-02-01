from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import shutil
import copy
from time import time
from datetime import timedelta
import h5py

'''
CHRONOS: population modeling of CRISPR readcount data
Joshua Dempster (dempster@broadinstitute.org)
The Broad Institute
'''

def write_hdf5(df, filename):
	if os.path.exists(filename):
		os.remove(filename)
	dest = h5py.File(filename)

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
				raise ValueError("The keys for %s (%r) do not match the other keys found (%r)" % (name, keys, set(entry.keys)))
		for key, val in entry.items():
			if not isinstance(val, pd.DataFrame):
				raise ValueError('expected Pandas dataframe for %s[%r]' %(name, key))

			if name == 'readcounts':
				assert val.index.duplicated().sum() == 0, "duplicated sequence IDs for readcounts %r" %key
				assert not val.isnull().all(axis=1).any(), \
						"All readcounts are null for one or more replicates in %s, please drop them" % key
				assert not val.isnull().all(axis=0).any(),\
						 "All readcounts are null for one or more guides in %s, please drop them" % key

			elif name == 'guide_gene_map':
				assert not guide_expected - set(val.columns), \
						"not all expected columns %r found for guide-gene map for %s. Found %r" %(guide_expected, key, val.columns) 
				assert val.sgrna.duplicated().sum() == 0, "duplicated sgRNAs for guide-gene map %r. Multiple gene alignments for sgRNAs are not supported." %key
				

			elif name == 'sequence_map':
				assert not sequence_expected - set(val.columns), \
						"not all expected columns %r found for sequence map for %s. Found %r" %(sequence_expected, key, val.columns)
				assert val.sequence_ID.duplicated().sum() == 0, "duplicated sequence IDs for sequence map %r" %key
				for batch in val.query('cell_line_name != "pDNA"').pDNA_batch.unique():
					assert batch in val.query('cell_line_name == "pDNA"').pDNA_batch.values, \
					"there are sequences with pDNA batch %s in library %s, but no pDNA measurements for that batch" %(batch, key)
				if val.days.max() > 50:
					print("\t\t\tWARNING: many days (%1.2f) found for %s.\n\t\t\tThis may cause numerical issues in fitting the model.\n\
					Consider rescaling all days by a constant factor so the max is less than 50." % (val.days.max(), key))
	
	for key in keys:
		if not readcounts is None and not sequence_map is None:
			assert not set(readcounts[key].index) - set(sequence_map[key].sequence_ID), \
				"\t\t\t mismatched sequence IDs between readcounts and sequence map for %r.\n\
				 Chronos expects `readcounts` to have guides as columns, sequence IDs as rows.\n\
				 Is your data transposed?" %key

		if not readcounts is None and not guide_gene_map is None:
			assert not set(readcounts[key].columns) - set(guide_gene_map[key].sgrna), \
				"mismatched map keys between readcounts and guide map for %s" % key


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


def calculate_fold_change(readcounts, sequence_map):
	'''
	Calculates fold change as the ratio of the RPM+1 of late time points to pDNA
	Parameters:
		readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns
		sequence_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"
	returns:
		fold_change (`pd.DataFrame`)
	'''
	check_inputs(readcounts={'default': readcounts}, sequence_map={'default': sequence_map})
	reps = sequence_map.query('cell_line_name != "pDNA"').sequence_ID
	pdna = sequence_map.query('cell_line_name == "pDNA"').sequence_ID
	rpm = pd.DataFrame(
		(1e6 * readcounts.values.T / readcounts.sum(axis=1).values + 1).T,
		index=readcounts.index, columns=readcounts.columns
	)
	fc = rpm.loc[reps]
	norm = rpm.loc[pdna].groupby(sequence_map.set_index('sequence_ID')['pDNA_batch']).median()
	try:
		fc = pd.DataFrame(fc.values/norm.loc[sequence_map.set_index('sequence_ID').loc[reps, 'pDNA_batch']].values,
			index=fc.index, columns=fc.columns
			)
	except Exception as e:
		print(fc.iloc[:3, :3],'\n')
		print(norm[:3], '\n')
		print(reps[:3], '\n')
		print(sequence_map[:3], '\n')
		raise e
	errors = []
	# if np.sum(fc.values <= 0) > 0:
	# 	errors.append("Fold change has zero or negative values:\n%r\n" % fc[fc <= 0].stack()[:10])
	# if (fc.min(axis=1) >= 1).any():
	# 	errors.append("Fold change has no values less than 1 for replicates\n%r" % fc.min(axis=1).loc[lambda x: x>= 1])
	if errors:
		raise RuntimeError('\n'.join(errors))
	return fc


def nan_outgrowths(readcounts, sequence_map, guide_gene_map, absolute_cutoff=2, gap_cutoff=2):
	'''
	NaNs readcounts in cases where all of the following are true:
		- The value  for the guide/replicate pair corresponds to the most positive log fold change of all guides and all replicates for a cell line
		- The logfold change for the guide/replicate pair is greater than `absolute_cutoff`
		- The difference between the lfc for this pair and the next most positive pair for that gene and cell line is greater than gap_cutoff
	Readcounts are mutated in place.
	Parameters:
		readcounts (`pandas.DataFrame`): readcount matrix with replicates on rows, guides on columns
		sequence_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"
		guide_gene_map (`pandas.DataFrame`): has string columns "sequence_ID", "cell_line_name", and "pDNA_batch"

	'''
	check_inputs(readcounts={'default': readcounts}, sequence_map={'default': sequence_map},
					 guide_gene_map={'default': guide_gene_map})
	print('calculating LFC')
	lfc = np.log2(calculate_fold_change(readcounts, sequence_map))


	print('finding maximum LFC cells')
	ggtemp = guide_gene_map.set_index('sgrna').gene.sort_index()
	sqtemp = sequence_map.set_index('sequence_ID').cell_line_name.sort_index()

	max_lfc = lfc.groupby(ggtemp, axis=1).max()
	potential_cols = max_lfc.columns[max_lfc.max() > absolute_cutoff]
	potential_rows= max_lfc.index[max_lfc.max(axis=1) > absolute_cutoff]
	max_lfc = max_lfc.loc[potential_rows, potential_cols]
	ggtemp = ggtemp[ggtemp.isin(potential_cols)]
	sqtemp = sqtemp[sqtemp.isin(potential_rows)]
	ggreversed = pd.Series(ggtemp.index.values, index=ggtemp.values).sort_index()
	sqreversed = pd.Series(sqtemp.index.values, index=sqtemp.values).sort_index()


	def second_highest(x):
		if len(x) == 1:
			return -np.inf
		return x.values[np.argpartition(-x.values, 1)[1]]


	max_row_2nd_column = lfc.T.groupby(ggtemp, axis=0).agg(second_highest).T  

	# print('constructing second of two second-highest matrices')
	# max_col_2nd_row = lfc.groupby(ggtemp, axis=1).max()\
	# 					.groupby(sqtemp, axis=0).agg(second_highest)    

	second_highest = max_row_2nd_column.loc[max_lfc.index, max_lfc.columns].values 
	# 	max_col_2nd_row.loc[max_lfc.index, max_lfc.columns].values
	# 	)
	gap = pd.DataFrame(max_lfc.values - second_highest, #second_highest
		index=max_lfc.index, columns=max_lfc.columns)

	print('finding sequences and guides with outgrowth')
	cases = max_lfc[(max_lfc > absolute_cutoff) & (gap > gap_cutoff)]
	cases = cases.stack()
	print('%i (%1.5f%% of) readcounts to be removed' % (
		len(cases), 
		100*len(cases)/np.product(readcounts.shape)
	))
	print(cases[:10])
	problems = pd.Series()
	for ind in cases.index:
		block = lfc.loc[ind[0], ggreversed.loc[[ind[1]]]]
		stacked = block[block == cases.loc[ind]]
		guide = stacked.index[0]
		problems.loc['%s&%s' % ind] = (ind[0], guide)


	print('NaNing bad outgrowths')
	for rep, guide in problems.values:
		readcounts.loc[rep, guide] = np.nan


	


##################################################################
#                M  O  D  E  L                                   #
##################################################################

class Chronos(object):
	'''
	Model class for inferring effect of gene knockout from readcount data. Takes in readcounts, mapping dataframes, and hyperparameters at init,
	then is trained with `train`. 

	Note on axes:

	Replicates and cell lines are always the rows/major axis of all dataframes and tensors. Guides and genes are always the columns/second axis.
	In cases where values vary per library, the object is a dict, and the library name is the key.

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

	Every set of parameters that are fit per-library are dicts. If `Chronos.v_a` is a dict, the subsequent attributes in the graph are also dicts.


	Settable Attributes: these CAN be set manually to interrogate the model or for other advanced uses, but NOT RECOMMENDED. Most users 
	will just want to read them out after training.
		guide_efficacy (`pandas.Series`): estimated on-target KO efficacy of reagents, between 0 and 1
		cell_efficacy (`dict` of `pandas.Series`): estimated cell line KO efficacy per library, between 0 and 1
		growth_rate (`dict` of `pandas.Series`): relative growth rate of cell lines, positive float. 1 is the average of all lines in lbrary.
		gene_effect ('pandas.DataFrame'): cell line by gene matrix of inferred change in growth rate caused by gene knockout
		screen_delay (`pandas.Series`): per gene delay between infection and appearance of growth rate phenotype
		initial_offset (`dict` of 'pandas.Series'): per sgrna estimated log fold pDNA error, per library. This value is exponentiated and mean-cented,
													then multiplied by the measured pDNA to infer the actual pDNA RPM of each guide.
													If there are fewer than 2 late time points, the mean of this value per gene is 0.
		days (`dict` of `pandas.Series`): number of days in culture for each replicate.
		learning_rate (`float`): current model learning rate. Will be overwritten when `train` is called.

	Unsettable (Calculated) Attributes:
		cost (`float`): the NB2 negative log-likelihood of the data under the current model, shifted to be 0 when the output RPM 
						perfectly matches the input RPM. Does not include regularization or terms involving only constants.
		cost_presum (`dict` of `pd.DataFrame`): the per-library, per-replicate, per-guide contribution to the cost.
		out (`dict` of `pd.DataFrame`): the per-library, per-replicate, per-guide model estimate of reads, unnormalized.
		output_norm (`dict` of `pandas.DataFrame`): `out` normalized so the sum of reads for each replicate is 1.
		efficacy (`pandas.DataFrame`): cell by guide efficacy matrix generated from the outer product of cell and guide efficacies
		initial (`dict` of `pandas.DataFrame`): estimated initial abundance of guides
		rpm (`dict` of `pandas.DataFrame`): the RPM of the measured readcounts / 1 million. Effectively a constant.
	'''

	default_timepoint_scale = .1 * np.log(2)
	default_cost_value = 0.67
	persistent_handles = set([])
	def __init__(self, 
				 readcounts,
				 #copy_number_matrix,
				 guide_gene_map,
				 sequence_map,
				 gene_effect_hierarchical=.1,
				 gene_effect_smoothing=.25,
				 kernel_width=5,
				 gene_effect_L1=0.1,
				 gene_effect_L2=0,
				 excess_variance=0.05,
				 guide_efficacy_reg=.5,
				 offset_reg=1,
				 growth_rate_reg=0.01,
				 smart_init=True,
				 cell_efficacy_guide_quantile=0.01,
				 initial_screen_delay=3,
				 scale_cost=0.67,
				 max_learning_rate=.02,
				 dtype=tf.double,
				 verify_integrity=True, 
				 log_dir=None,
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

			gene_effect_hierarchical (`float`): regularization of individual gene effect scores towards the mean across cell lines
			gene_effect_smoothing (`float`): regularization of individual gene scores towards mean after Gaussian kernel convolution
			kernel_width (`float`): width (SD) of the Gaussian kernel for the smoothing regularization
			gene_effect_L1 (`float`): regularization of gene effect CELL LINE MEAN towards zero with L1 penalty
			gene_effect_L2 (`float`): regularization of individual gene scores towards zero with L2 penalty
			offset_reg (`float`): regularization of pDNA error
			growth_rate_reg (`float`): regularization of the negative log of the relative growth rate
			guide_efficacy_reg (`float`): regularization of the gap between the two strongest guides' efficacy per gene,
										or of the gap between them and 1 if only one late timepoint is present in readcounts for that library
			excess_variance (`float` or `dict`): measure of Negative Binomial overdispersion for the cost function, 
												overall or per cell line and library. 
			max_learning_rate (`float`): passed to AdamOptimizer after initial burn-in period during training
			verify_integrity (`bool`): whether to check each itnermediate tensor computed by Chronos for innappropriate values
			log_dir (`path` or None): if provided, location where Tensorboard snapshots will be saved
			cell_efficacy_init (`bool`): whether to initialize cell efficacies using the fold change of the most depleted guides 
										at the last timepoint
			celll_efficacy_guide_quantile (`float`): quantile of guides to use to estimate cell screen efficacy. Between 0 and 0.5.
			initial_screen_delay (`float`): how long after infection before growth phenotype kicks in, in days. If there are fewer than
											3 late timepoints this initial value will be left unchanged.
			dtype (`tensorflow.double` or `tensorflow.float`): numerical precision of the computation. Strongly recommend to leave this unchanged.
			scale_cost (`bool`): The likelihood cost will be scaled to always be initially this value (default 0.67) for all data. 
								This encourages more consistent behavior across datasets when leaving the other regularization hyperparameters 
								constant. Pass 0, False, or None to avoid cost scaling.
		'''


		###########################    I N I T I A L      C  H  E  C  K  S  ############################

		check_inputs(readcounts=readcounts, sequence_map=sequence_map, guide_gene_map=guide_gene_map)
		sequence_map = self._make_pdna_unique(sequence_map, readcounts)
		excess_variance = self._check_excess_variance(excess_variance, readcounts, sequence_map)
		self.np_dtype = {tf.double: np.float64, tf.float32: np.float32}[dtype]
		self.keys = list(readcounts.keys())
		if scale_cost:
			try:
				scale_cost = float(scale_cost)
				assert 0 < scale_cost, "scale_cost must be positive"
			except:
				raise ValueError("scale_cost must be None, False, or a semi-positive number")


		####################    C  R  E  A  T  E       M  A  P  P  I  N  G  S   ########################

		(self.guides, self.genes, self.all_guides, self.all_genes,
			self.guide_map, self.column_map
			) = self._get_column_attributes(readcounts, guide_gene_map)

		(self.sequences, self.pDNA_unique, self.cells, self.all_sequences, \
			self.all_cells, self.cell_indices, self.replicate_map, self.index_map, 
			self.line_index_map, self.batch_map
			) = self._get_row_attributes(readcounts, sequence_map)


		##################    A  S  S  I  G  N       C  O  N  S  T  A  N  T  S   #######################

		print('\n\nassigning float constants')
		self.guide_efficacy_reg = float(guide_efficacy_reg)
		self.gene_effect_L1 = float(gene_effect_L1)
		self.gene_effect_L2 = float(gene_effect_L2)
		self.gene_effect_hierarchical = float(gene_effect_hierarchical)
		self.growth_rate_reg = float(growth_rate_reg)
		self.offset_reg = float(offset_reg)
		self.gene_effect_smoothing = float(gene_effect_smoothing)
		self.kernel_width = float(kernel_width)
		self.cell_efficacy_guide_quantile = float(cell_efficacy_guide_quantile)
		if not 0 < self.cell_efficacy_guide_quantile < .5:
			raise ValueError("cell_efficacy_guide_quantile should be greater than 0 and less than 0.5")

		self.nguides, self.ngenes, self.nlines, self.nsequences = (
			len(self.all_guides), len(self.all_genes), len(self.all_cells), len(self.all_sequences)
		)

		self._excess_variance = self._get_excess_variance_tf(excess_variance)
		self.median_timepoint_counts = self._summarize_timepoint(sequence_map, np.median)

		self._initialize_graph(max_learning_rate, dtype)

		self._gene_effect_mask, self.mask_count = self._get_gene_effect_mask(dtype)
		self._days = self._get_days(sequence_map, dtype)
		self._rpm, self._mask = self._get_late_tf_timepoints(readcounts, dtype)

		self._measured_initial = self._get_tf_measured_initial(readcounts, sequence_map, dtype)


		##################    C  R  E  A  T  E       V  A  R  I  A  B  L  E  S   #######################

		print('\n\nBuilding variables')

		(self.v_initial, self._initial_core, 
			self._initial, self._initial_offset, self._grouped_initial_offset) = self._get_initial_tf_variables(dtype)

		(self.v_guide_efficacy, self._guide_efficacy) = self._get_tf_guide_efficacy(dtype)

		(self.v_growth_rate, self._growth_rate, self._line_presence_boolean) = self._get_tf_growth_rate(dtype)

		(self.v_cell_efficacy, self._cell_efficacy) = self._get_tf_cell_efficacy(dtype)

		(self.v_screen_delay, self._screen_delay) = self._get_tf_screen_delay(initial_screen_delay, dtype)

		(self.v_mean_effect, self.v_residue, self._residue, self._true_residue, self._combined_gene_effect
			) = self._get_tf_gene_effect(dtype)


		#############################    C  O  R  E      M  O  D  E  L    ##############################

		print("\n\nConnecting graph nodes in model")

		self._effective_days = self._get_effect_days(self._screen_delay, self._days)

		self._gene_effect_growth = self._get_gene_effect_growth(self._combined_gene_effect, self._growth_rate)

		self._efficacy, self._selected_efficacies = self._get_combined_efficacy(self._cell_efficacy,self. _guide_efficacy)

		self._growth, self._change = self._get_growth_and_fold_change(self._gene_effect_growth, self._effective_days, 
																		self._selected_efficacies)

		self._out, self._output_norm = self._get_abundance_estimates(self._initial, self._change)


		#####################################    C  O  S  T    #########################################

		print("\n\nBuilding all costs")

		self._total_guide_reg_cost = self._get_guide_regularization(self._guide_efficacy, dtype)

		self._smoothed_presum = self._get_smoothed_ge_regularization(self.v_mean_effect, self._true_residue, kernel_width, dtype)

		self._initial_cost = self._get_initial_regularization(self._initial_offset)
			
		self._cost_presum, self._cost, self._scale = self._get_nb2_cost(self._excess_variance, self._output_norm, self._rpm, self._mask,
			dtype)
		self.run_dict.update({self._scale: 1.0})

		self._full_cost = self._get_full_cost(dtype)


		#########################    F  I  N  A  L  I  Z  I  N  G    ###################################

		print('\nCreating optimizer')		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

		default_var_list = [
				self.v_mean_effect,
				self.v_residue, 
				self.v_guide_efficacy,
				self.v_initial,
				self.v_growth_rate
				]
		# if all([val > 2 for val in self.median_timepoint_counts.values()]):
		# 	"All libraries have sufficient timepoints to estimate screen_delay, adding to estimate"
		# 	default_var_list.append(self.v_screen_delay)

		self._ge_only_step = self.optimizer.minimize(self._full_cost, var_list=[self.v_mean_effect, self.v_residue])
		self._step = self.optimizer.minimize(self._full_cost, var_list=default_var_list)
		self._merged = tf.summary.merge_all()

		if log_dir is not None:
			print("\tcreating log at %s" %log_dir)
			if os.path.isdir(log_dir):
				shutil.rmtree(log_dir)
			os.mkdir(log_dir)
			self.log_dir = log_dir
			self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
		
		init_op = tf.global_variables_initializer()
		print('initializing variables')
		self.sess.run(init_op)

		if scale_cost:
			denom = self.cost
			self.run_dict.update({self._scale: scale_cost/denom})

		if smart_init:
			print("estimating initial screen efficacy")
			self.smart_initialize(readcounts, sequence_map, cell_efficacy_guide_quantile)

		if verify_integrity:
			print("\tverifying graph integrity")
			self.nan_check()

		self.epoch = 0

		print('ready to train')



    ################################################################################################
	##############   I N I T I A L I Z A T I O N    M  E  T  H  O  D  S    #########################
	################################################################################################


	def get_persistent_input(self, dtype, data, name=''):
		placeholder = tf.placeholder(dtype=dtype, shape=data.shape)
		# Persistent tensor to hold the data in tensorflow. Helpful because TF doesn't allow 
		# graph definitions larger than 2GB (so can't use constants), and passing the feed dict each time is slow.
		# This feature is poorly documented, but the handle seems to refer not to a tensor but rather a tensor "state" -
		# the state of a placeholder that's been passed the feed dict. This is what persists. Annoyingly, it then becomes
		# impossible to track the shape of the tensor.
		state_handle = self.sess.run(tf.get_session_handle(placeholder), {placeholder: data})
		# why TF's persistence requires two handles, I don't know. But it does.
		tensor_handle, data = tf.get_session_tensor(state_handle.handle, dtype=dtype, name=name)
		self.run_dict[tensor_handle] = state_handle.handle
		self.persistent_handles.add(state_handle.handle)
		return data


###########################    I N I T I A L      C  H  E  C  K  S  ############################

	def _make_pdna_unique(self, sequence_map, readcounts):
		#guarantee unique pDNA batches
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
				assert isinstance(val, pd.Series), "the excess_variance values provided for the different datasets must be pandas.Series objects, not\n%r" % val
				diff = set(val.index) ^ set(sequence_map[key].cell_line_name)
				assert len(diff) < 2, "difference between index values\n%r\nfor excess_variance and cell lines found in %s" % (diff, key)
		return excess_variance


####################    C  R  E  A  T  E       M  A  P  P  I  N  G  S   ########################

	def make_map(melted_map, outer_list, inner_list, dtype=np.float64):
		'''
		takes a sorted list of indices, targets, and a pd.Series that maps between them and recomputes the mapping between them
		as two arrays of integer indices suitable for gather function calls. 
		The mapping can only include a subset of either the outer or inner list and vice versa.
		The mapping's indices must be unique.
		'''
		melted_map = melted_map[melted_map.index.isin(outer_list) & melted_map.isin(inner_list)]
		outer_array = np.array(outer_list)
		gather_outer = np.searchsorted(outer_array, melted_map.index).astype(np.int)
		inner_array = np.array(inner_list)
		gather_inner = np.searchsorted(inner_array, melted_map.values).astype(np.int)
		args = { 
			'gather_ind_inner': gather_inner, 
			'gather_ind_outer': gather_outer}
		return args


	def _get_column_attributes(self, readcounts, guide_gene_map):
		print('\n\nFinding all unique guides and genes')
		#guarantees the same sequence of guides and genes within each library
		guides = {key: val.columns for key, val in readcounts.items()}
		genes = {key: val.set_index('sgrna').loc[guides[key], 'gene'] for key, val in guide_gene_map.items()}
		all_guides = sorted(set.union(*[set(v) for v in guides.values()]))
		all_genes = sorted(set.union(*[set(v.values) for v in genes.values()]))
		for key in self.keys:
			print("found %i unique guides and %i unique genes in %s" %(
				len(set(guides[key])), len(set(genes[key])), key
				))
		print("found %i unique guides and %i unique genes overall" %(len(all_guides), len(all_genes)))
		print('\nfinding guide-gene mapping indices')
		guide_map = {key: 
				Chronos.make_map(guide_gene_map[key][['sgrna', 'gene']].set_index('sgrna').iloc[:, 0],
				 all_guides, all_genes, self.np_dtype)
				for key in self.keys}
		column_map = {key: np.array(all_guides)[guide_map[key]['gather_ind_outer']]
								for key in self.keys}
		return guides, genes, all_guides, all_genes, guide_map, column_map


	def _get_row_attributes(self, readcounts, sequence_map):
		print('\nfinding all unique sequenced replicates, cell lines, and pDNA batches')
		#guarantees the same sequence of sequence_IDs and cell lines within each library.
		sequences = {key: val[val.cell_line_name != 'pDNA'].sequence_ID for key, val in sequence_map.items()}
		pDNA_batches = {key: list(val[val.cell_line_name != 'pDNA'].pDNA_batch.values)
								for key, val in sequence_map.items()}
		pDNA_unique = {key: sorted(set(val)) for key, val in pDNA_batches.items()}
		cells = {key: val[val.cell_line_name != 'pDNA']['cell_line_name'].unique() for key, val in sequence_map.items()}
		all_sequences = sorted(set.union(*tuple([set(v.values) for v in sequences.values()])))
		all_cells = sorted(set.union(*tuple([set(v) for v in cells.values()])))
		#This is necessary consume copy number provided for only the cell-guide blocks present in each library
		cell_indices = {key: [all_cells.index(s) for s in v] 
									for key, v in cells.items()}

		assert len(all_sequences) == sum([len(val) for val in sequences.values()]
			), "sequence IDs must be unique among all datasets"
		for key in self.keys:
			print("found %i unique sequences (excluding pDNA) and %i unique cell lines in %s" %(
				len(set(sequences[key])), len(set(cells[key])), key
				))
		print("found %i unique replicates and %i unique cell lines overall" %(len(all_sequences), len(all_cells)))

		print('\nfinding replicate-cell line mappings indices')
		replicate_map = {key: 
				Chronos.make_map(sequence_map[key][['sequence_ID', 'cell_line_name']].set_index('sequence_ID').iloc[:, 0],
				 all_sequences, all_cells, self.np_dtype)
				for key in self.keys}
		index_map = {key: np.array(all_sequences)[replicate_map[key]['gather_ind_outer']]
								for key in self.keys}
		line_index_map = {key: np.array(all_cells)[replicate_map[key]['gather_ind_inner']]
								for key in self.keys}

		print('\nfinding replicate-pDNA mappings indices')
		batch_map = {key: 
				Chronos.make_map(sequence_map[key][['sequence_ID', 'pDNA_batch']].set_index('sequence_ID').iloc[:, 0],
				 all_sequences, pDNA_unique[key], self.np_dtype)
				for key in self.keys}

		return sequences, pDNA_unique, cells, all_sequences, all_cells, cell_indices, replicate_map, index_map, line_index_map, batch_map


##################    A  S  S  I  G  N       C  O  N  S  T  A  N  T  S   #######################

	def _get_excess_variance_tf(self, excess_variance):
		_excess_variance = {}
		for key in self.keys:
				try:
					_excess_variance[key] = tf.constant(excess_variance[key][self.line_index_map[key]].values.reshape((-1, 1)))
				except IndexError:
					raise IndexError("difference between index values for excess_variance and cell lines found in %s" % key)
				except TypeError:
					_excess_variance[key] = tf.constant(excess_variance * np.ones(shape=(len(self.line_index_map[key]), 1)))
		return _excess_variance


	def _summarize_timepoint(self, sequence_map, func):
		out = {}
		for key, val in sequence_map.items():
			out[key] = func(val.groupby("cell_line_name").days.agg(lambda v: len(v.unique())).drop('pDNA').values)
		return out



	def _initialize_graph(self, max_learning_rate, dtype):
		print('initializing graph')
		self.sess = tf.Session()
		self._learning_rate = tf.placeholder(shape=tuple(), dtype=dtype)
		self.run_dict = {self._learning_rate: max_learning_rate}
		self.max_learning_rate = max_learning_rate
		self.persistent_handles = set([])


	def _get_gene_effect_mask(self, dtype):
		print('\nbuilding gene effect mask')
		mask = pd.DataFrame(0, index=self.all_cells, columns=self.all_genes, dtype=np.bool)
		for cell in self.all_cells:
			libraries = [key for key in self.keys if cell in self.cells[key]]
			covered_genes = sorted(set.intersection(*[set(self.genes[key]) for key in libraries]))
			mask.loc[cell, covered_genes] = 1
		_gene_effect_mask = tf.constant(mask.astype(self.np_dtype).values, dtype=dtype)
		mask_count = (mask == 1).sum().sum()
		print('made gene_effect mask, excluded %i (%1.5f) values' % ((mask == 0).sum().sum(), (mask == 0).mean().mean()))
		return _gene_effect_mask, mask_count


	def _get_days(self, sequence_map, dtype):	
		print('\nbuilding doubling vectors')
		_days = {key: 
			tf.constant(Chronos.default_timepoint_scale * val.set_index('sequence_ID').loc[self.index_map[key]].days.astype(self.np_dtype).values, 
				dtype=dtype, shape=(len(self.index_map[key]), 1), name="days_%s" % key)
			for key, val in sequence_map.items()}
		for key in self.keys:
			print("made days vector of shape %r for %s" %(
				_days[key].get_shape().as_list(), key))
		return _days


	def _get_late_tf_timepoints(self, readcounts, dtype):
		print("\nbuilding late observed timepoints")
		_rpm = {}
		_mask = {}
		for key in self.keys:
			rpm_np = readcounts[key].loc[self.index_map[key], self.column_map[key]].copy()
			rpm_np = 1e6 * (rpm_np.values + 1e-32) / (rpm_np.fillna(0).values + 1e-32).sum(axis=1).reshape((-1, 1))
			mask = pd.notnull(rpm_np)
			_mask[key] = tf.constant(mask, dtype=tf.bool, name='NaN_mask_%s' % key)
			rpm_np[~mask] = 0
			_rpm[key] = self.get_persistent_input(dtype, rpm_np, name='rpm_%s' % key)
			print("\tbuilt normalized timepoints for %s with shape %r (replicates X guides)" %(
				key, rpm_np.shape))
		return _rpm, _mask


	def _get_tf_measured_initial(self, readcounts, sequence_map, dtype):
		print('\nbuilding initial reads')
		_measured_initial = {}

		for key in self.keys:
			rc = readcounts[key]
			sm = sequence_map[key]
			sm = sm[sm.cell_line_name == 'pDNA']
			batch = rc.loc[sm.sequence_ID]
			if batch.empty:
				raise ValueError("No sequenced entities are labeled 'pDNA', or there are no readcounts for those that are")
			if batch.shape[0] > 1:
				batch = batch.groupby(sm.pDNA_batch.values).sum().astype(self.np_dtype)
			else:
				batch = pd.DataFrame({self.pDNA_unique[key][0]: batch.iloc[0]}).T.astype(self.np_dtype)
			batch = batch.loc[self.pDNA_unique[key], self.column_map[key]]
			if batch.isnull().sum().sum() != 0:
				print(batch)
				raise RuntimeError("NaN values encountered in batched pDNA")
			initial_normed = batch.divide(batch.sum(axis=1), axis=0).values + 1e-8
			_measured_initial[key] = tf.constant(initial_normed, name='measured_initial_%s' % key, dtype=dtype)
		return _measured_initial


##################    C  R  E  A  T  E       V  A  R  I  A  B  L  E  S   #######################

	def _get_initial_tf_variables(self, dtype):
		print("\nbuilding initial reads estimate")

		v_initial = {}
		_initial_core = {}
		_initial = {}
		_initial_offset = {}
		_grouped_initial_offset = {}
		for key in self.keys:
			initial_normed = self.sess.run(self._measured_initial[key], self.run_dict)
			v_initial[key] = tf.Variable(np.zeros((initial_normed.shape[1], 1), dtype=self.np_dtype), dtype=dtype, name='initial_%s' % key)
			_initial_offset[key] = tf.exp(v_initial[key] - tf.reduce_mean(v_initial[key]))


			_grouped_initial_offset[key] = tf.transpose(tf.math.unsorted_segment_mean(
					_initial_offset[key],
					self.guide_map[key]['gather_ind_inner'],
					num_segments=self.ngenes,
					name='grouped_diff_%s' % key
			))

			_initial_core[key] = self._measured_initial[key] *\
				tf.exp(tf.transpose(_initial_offset[key]) - tf.gather(_grouped_initial_offset[key], 
					self.guide_map[key]['gather_ind_inner'], axis=1))
			

			_initial[key] = tf.gather(_initial_core[key] / tf.reshape(tf.reduce_sum(_initial_core[key], axis=1), shape=(-1, 1)),
					self.batch_map[key]['gather_ind_inner'], 
					axis=0, 
					name='initial_read_est_%s' % key
					)


			print("made initial batch with shape %r for %s" %(
				initial_normed.shape, key))

		return v_initial, _initial_core, _initial, _initial_offset, _grouped_initial_offset


	def _get_tf_guide_efficacy(self, dtype):		
		print("building guide efficacy")
		v_guide_efficacy = tf.Variable(
			#last guide is dummy
			tf.random_normal(shape=(1, self.nguides+1), stddev=.01, dtype=dtype),
								name='guide_efficacy_base', dtype=dtype)
		_guide_efficacy = tf.exp(-tf.abs(v_guide_efficacy), name='guide_efficacy')
		tf.summary.histogram("guide_efficacy", _guide_efficacy)
		print("built guide efficacy: shape %r" %_guide_efficacy.get_shape().as_list())
		return v_guide_efficacy, _guide_efficacy


	def _get_tf_growth_rate(self, dtype):
		print("building growth rate")
		v_growth_rate = { key: tf.Variable(
				tf.random_normal(shape=(self.nlines, 1), stddev=.01, mean=1, dtype=dtype),
								name='growth_rate_base_%s' % key, dtype=dtype)
				for key in self.keys}
		_line_presence_mask = {key: tf.constant( np.array([s in self.cells[key] for s in self.all_cells], dtype=self.np_dtype).reshape((-1, 1)) )
										for key in self.keys}
		_line_presence_boolean = {key: tf.constant( np.array([s in self.cells[key] for s in self.all_cells], dtype=np.bool), dtype=tf.bool)
										for key in self.keys}
		_growth_rate_square = {key: (val * _line_presence_mask[key]) ** 2 for key, val in v_growth_rate.items()}
		_growth_rate = {key: tf.divide(val, tf.reduce_mean(tf.boolean_mask(val, _line_presence_boolean[key])), 
								name="growth_rate_%s" % key)
							for key, val in _growth_rate_square.items()}
		print("built growth rate: shape %r" % {key: val.get_shape().as_list() 
			for key, val in _growth_rate.items()})
		return v_growth_rate, _growth_rate, _line_presence_boolean


	def _get_tf_cell_efficacy(self, dtype):
		print("\nbuilding cell line efficacy")
		v_cell_efficacy = { key: tf.Variable(
				tf.random_normal(shape=(self.nlines, 1), stddev=.01, mean=0, dtype=dtype),
								name='cell_efficacy_base_%s' % key, dtype=dtype)
				for key in self.keys}
		_cell_efficacy = {key: tf.exp(-tf.abs(v_cell_efficacy[key]),
						  name='cell_efficacy_%s' % key)
				for key in self.keys}
		print("built cell line efficacy: shapes %r" % {key: v.get_shape().as_list() for key, v in _cell_efficacy.items()})
		return v_cell_efficacy, _cell_efficacy


	def _get_tf_screen_delay(self, initial_screen_delay, dtype):
		print("building screen delay")
		v_screen_delay = tf.Variable(np.sqrt(Chronos.default_timepoint_scale * initial_screen_delay) * np.ones((1, self.ngenes), dtype=self.np_dtype),
				 dtype=dtype)
		_screen_delay = tf.square(v_screen_delay, name="screen_delay")
		tf.summary.histogram("screen_delay", _screen_delay)
		print("built screen delay")
		return v_screen_delay, _screen_delay


	def _get_tf_gene_effect(self, dtype):
		print("building gene effect")
		gene_effect_est = np.random.uniform(-.0001, .00005, size=(self.nlines, self.ngenes)).astype(self.np_dtype)
		#self._combined_gene_effect = tf.Variable(gene_effect_est, dtype=dtype, name="Gene_Effect")
		v_mean_effect = tf.Variable(np.random.uniform(-.0001, .00005, size=(1, self.ngenes)), name='GE_mean', dtype=dtype)
		v_residue = tf.Variable(gene_effect_est, dtype=dtype, name='GE_deviation') 
		_residue = v_residue * self._gene_effect_mask
		_true_residue =  (
				v_residue - (tf.reduce_sum(v_residue, axis=0)/tf.reduce_sum(self._gene_effect_mask, axis=0) )[tf.newaxis, :]
			) * self._gene_effect_mask
		_combined_gene_effect = v_mean_effect + _true_residue
		tf.summary.histogram("mean_gene_effect", v_mean_effect)
		print("built core gene effect: %i cell lines by %i genes" %tuple(_combined_gene_effect.get_shape().as_list()))
		return v_mean_effect, v_residue, _residue, _true_residue, _combined_gene_effect


#############################    C  O  R  E      M  O  D  E  L    ##############################

	def _get_effect_days(self, _screen_delay, _days):
		print("\nbuilding effective days")
		with tf.name_scope("days"):
			_effective_days = {key: 
				tf.clip_by_value(val - _screen_delay, 0, 100)
			for key, val in _days.items()}

		print("built effective days, shapes %r" % {key: val.get_shape().as_list() for key, val in _effective_days.items()})
		return _effective_days


	def _get_gene_effect_growth(self, _combined_gene_effect, _growth_rate):
		print('\nbuilding gene effect growth graph nodes')
		with tf.name_scope('GE_G'):
			_gene_effect_growth = {key: _combined_gene_effect * _growth_rate[key]
								for key in self.keys}
		print("built gene effect growth graph nodes, shapes %r" % {key: val.get_shape().as_list() 
			for key, val in _gene_effect_growth.items()})
		return _gene_effect_growth


	def _get_combined_efficacy(self, _cell_efficacy, _guide_efficacy):
		print('\nbuilding combined efficacy')
		with tf.name_scope('efficacy'):
			_efficacy = {key: 
					tf.matmul(_cell_efficacy[key], tf.gather(_guide_efficacy, self.guide_map[key]['gather_ind_outer'], axis=1, name='guide'),
				 name="combined")
				 for key in self.keys} #cell line by all guide matrix
			_selected_efficacies = {
				key: tf.gather(#expand to replicates in given library
						_efficacy[key], 
						self.replicate_map[key]['gather_ind_inner'],
						name="replicate"
						)
				for key in self.keys
			}
		print("built combined efficacy, shape %r" % {key: v.get_shape().as_list()for key, v in _efficacy.items()})
		print("built expanded combined efficacy, shapes %r" % {key: val.get_shape().as_list() for key, val in _selected_efficacies.items()})
		return _efficacy, _selected_efficacies


	def _get_growth_and_fold_change(self, _gene_effect_growth, _effective_days, _selected_efficacies):
		print("\nbuilding growth estimates of edited cells and overall estimates of fold change in guide abundance")
		_change = {}
		_growth = {}
		with tf.name_scope("FC"):
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
		print("built growth and change")
		return _growth, _change


	def _get_abundance_estimates(self, _initial, _change):
		print("\nbuilding unnormalized estimates of final abundance")
		_out = {key: tf.multiply(_initial[key], _change[key], name="out_%s" % key)
				 for key in self.keys}
		
		print("built unnormalized abundance")

		print("\nbuilding normalized estimates of final abundance")
		with tf.name_scope('out_norm'):
			_output_norm = {key: 
					1e6 * tf.divide((val + 1e-32), tf.reshape(tf.reduce_sum(val, axis=1), shape=(-1, 1)),
						name=key
						)
							for key, val in _out.items()}
		print("built normalized abundance")
		return _out, _output_norm


#####################################    C  O  S  T    #########################################

	def _get_guide_regularization(self, _guide_efficacy, dtype):
		print('\nassembling guide efficacy regularization')
		guide_map_pd = {}
		max_guides = {}
		_guide_reg_matrix = {}
		_guide_reg_cost = {}

		#guarantee that the "dummy" guide will never be the second most effective guide for any gene
		fixed_adjust_np = np.zeros(self.nguides+1, dtype=self.np_dtype)
		fixed_adjust_np[-1] = 1e6
		_guide_reg_base = 1.0 / _guide_efficacy[0] + tf.constant(fixed_adjust_np, dtype=dtype)
		lists = []
		for key in self.keys:
			guide_map_pd[key] = pd.Series(self.guide_map[key]['gather_ind_outer'], 
										index=self.guide_map[key]['gather_ind_inner']
									).sort_index()
			ngenes = guide_map_pd[key].index.nunique()
			value_counts = pd.Series(self.guide_map[key]['gather_ind_inner']).value_counts()
			max_guides[key] = value_counts.max()
			nextras = sum(max_guides[key] - value_counts)
			index = [gene for gene in guide_map_pd[key].index.unique() for i in range(max_guides[key] - value_counts[gene])]
			dummies = pd.Series([self.nguides] * nextras, index=index)
			guide_map_pd[key] = pd.concat([guide_map_pd[key], dummies]).sort_index()

			reg_matrix_ind = guide_map_pd[key].values.reshape((ngenes, max_guides[key])).astype(np.int)
			_guide_reg_matrix[key] = tf.contrib.framework.sort(
				tf.gather(_guide_reg_base, reg_matrix_ind, axis=0),
				direction='ASCENDING',
				name='sorted_guide_reg_%s' % key
			)
			if False:#all([val > 2 for val in self.median_timepoint_counts.values()]):
				# only regularize gap between first and second most efficacious guide
				_guide_reg_cost[key] = tf.reduce_mean(
					_guide_reg_matrix[key][:, 1] - _guide_reg_matrix[key][:, 0],
					name="guide_reg_cost_%s" % key
				)
			else:
				#regularize total of first and second most efficacious guide - at least two guides must be near 1
				_guide_reg_cost[key] = tf.reduce_mean(
					_guide_reg_matrix[key][:, 1] + _guide_reg_matrix[key][:, 0],
					name="guide_reg_cost_%s" % key
				)

		_total_guide_reg_cost = 1.0/len(_guide_reg_cost) * tf.add_n(list(_guide_reg_cost.values()))
		return _total_guide_reg_cost


	def _get_smoothed_ge_regularization(self, v_mean_effect, _true_residue, kernel_width, dtype):
		print("building smoothed regularization")
		kernel_size = int(6 * kernel_width)
		kernel_size = kernel_size + kernel_size % 2 + 1 #guarantees odd width
		kernel = np.exp( -( np.arange(kernel_size, dtype=self.np_dtype) - kernel_size//2 )**2/ (2*kernel_width**2) )
		kernel = kernel / kernel.sum()
		_kernel = tf.constant(kernel, dtype=dtype, name='kernel')[:, tf.newaxis, tf.newaxis]
		_ge_argsort = tf.argsort(v_mean_effect[0])
		_residue_sorted = tf.gather(_true_residue, _ge_argsort, axis=1)[:, :, tf.newaxis]
		_residue_smoothed = tf.nn.convolution(_residue_sorted, _kernel, padding='SAME')
		_smoothed_presum = tf.square(_residue_smoothed)
		return _smoothed_presum


	def _get_initial_regularization(self, _initial_offset):
		print("\nbuilding initial reads regularization/cost")
		_initial_cost = {key:
			tf.reduce_mean( tf.square(_initial_offset[key]), 
				name='cost_initial_%s' %key)
			for key in self.keys
		}
		return _initial_cost


	def _get_nb2_cost(self, _excess_variance, _output_norm, _rpm, _mask, dtype):
		print('\nbuilding NB2 cost')
		
		with tf.name_scope('cost'):
			# the NB2 cost: (yi + 1/alpha) * ln(1 + alpha mu_i) - yi ln(alpha mu_i)
			# modified with constants and -mu_i - which makes it become the multinomial cost in the limit alpha -> 0
			_cost_presum = {key: 
								 	(
								 		((_rpm[key]+1e-6) + 1./_excess_variance[key]) * tf.log(
								 			(1 + _excess_variance[key] * (_output_norm[key] + 1e-6)) /
								 			(1 + _excess_variance[key] * (_rpm[key] + 1e-6))
								 	) +
									(_rpm[key]+1e-6) * tf.log((_rpm[key] + 1e-6) / (_output_norm[key] + 1e-6) ) 
									)
								for key in self.keys}

			_scale = tf.placeholder(dtype=dtype, shape=(), name='scale')
			_cost =  _scale/len(self.keys) * tf.add_n([tf.reduce_mean(tf.boolean_mask(v, _mask[key]))
			 										 for key, v in _cost_presum.items()]
			 			)

			tf.summary.scalar("unregularized_cost", _cost)
			return _cost_presum, _cost, _scale


	def _get_full_cost(self, dtype):
		print("building other regularizations")
		with tf.name_scope('full_cost'):
			self._L1_penalty = self.gene_effect_L1 * tf.square(tf.reduce_sum(self._combined_gene_effect)/self.mask_count) 
			self._L2_penalty = self.gene_effect_L2 * tf.reduce_sum(tf.square(self._combined_gene_effect))/self.mask_count
			self._hier_penalty = self.gene_effect_hierarchical * tf.reduce_sum(tf.square(self._true_residue))/self.mask_count
			self._growth_reg_cost = -self.growth_rate_reg * 1.0/len(self.keys) * tf.add_n([
													tf.reduce_mean( tf.log(tf.boolean_mask(v, self._line_presence_boolean[key])) )
													for key, v in self._growth_rate.items()
																					])

			self._guide_efficacy_reg = tf.placeholder(dtype, shape=())
			self.run_dict[self._guide_efficacy_reg] = self.guide_efficacy_reg

			self._guide_reg_cost = self._guide_efficacy_reg * self._total_guide_reg_cost
			self._smoothed_cost = self.gene_effect_smoothing * tf.reduce_mean(self._smoothed_presum)

			self._offset_reg = tf.placeholder(dtype, shape=())
			self.run_dict[self._offset_reg] = self.offset_reg
			self._initial_cost_sum = self._offset_reg * 1.0/len(self.keys) * tf.add_n(list(self._initial_cost.values()))


			_full_cost = self._cost + \
							self._L1_penalty + self._L2_penalty + self._guide_reg_cost + self._hier_penalty  + \
							self._growth_reg_cost + self._initial_cost_sum + \
							self._smoothed_cost 

			tf.summary.scalar("L1_penalty", self._L1_penalty)
			tf.summary.scalar("L2_penalty", self._L2_penalty)
			tf.summary.scalar("hierarchical_penalty", self._hier_penalty)
		return _full_cost


#########################    F  I  N  A  L  I  Z  I  N  G    ###################################

	def cell_efficacy_estimate(self, fold_change, sequence_map, last_reps, cell_efficacy_guide_quantile=.01):
		'''
		Estimate the maximum depletion possible in cell lines as the lowest X percentile guide fold-change in
		the last timepoint measured. Multiple replicates for a cell line at the same last timepoint are median-collapsed
		before the percentile is measured.
		'''
		fc = fold_change.loc[last_reps].groupby(sequence_map.set_index('sequence_ID').cell_line_name).median()
		cell_efficacy = 1 - fc.quantile(cell_efficacy_guide_quantile, axis=1)
		if (cell_efficacy <=0 ).any() or (cell_efficacy > 1).any() or cell_efficacy.isnull().any():
			raise RuntimeError("estimated efficacy outside bounds. \n%r\n%r" % (cell_efficacy.sort_values(), fc))

		return cell_efficacy


	def smart_initialize(self, readcounts, sequence_map, cell_efficacy_guide_quantile):
		cell_eff_est = {}
		for key in self.keys:
			print('\t', key)
			sm = sequence_map[key]
			last_reps = extract_last_reps(sm)
			fc = calculate_fold_change(readcounts[key], sm)
			cell_eff_est[key] = self.cell_efficacy_estimate(fc, sm, last_reps, cell_efficacy_guide_quantile)
		self.cell_efficacy = cell_eff_est


	def nan_check(self):
		#labeled data
		print('verifying user inputs')
		for key in self.keys:
			if pd.isnull(self.sess.run(self._days[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._days[%s]" %key
			if pd.isnull(self.sess.run(self._initial[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._initial[%s]" %key
			if pd.isnull(self.sess.run(self._rpm[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._excess_variance[%s]" %key
			if (self.sess.run(self._excess_variance[key], self.run_dict) < 0).sum().sum() > 0:
				assert False, "negative values found in self._excess_variance[%s]" %key

		#variables
		print('verifying variables')
		if pd.isnull(self.sess.run(self._combined_gene_effect, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self._combined_gene_effect"
		if pd.isnull(self.sess.run(self.v_guide_efficacy, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self.v_guide_efficacy"
		if pd.isnull(self.sess.run(self._guide_efficacy, self.run_dict)).sum().sum() > 0:
			assert False, "nulls found in self._guide_efficacy"

		#calculated terms
		print('verifying calculated terms')
		for key in self.keys:
			if pd.isnull(self.sess.run(self.v_cell_efficacy[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self.v_cell_efficacy[%r]" % key
			if pd.isnull(self.sess.run(self._efficacy[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._efficacy[%r]" % key
			print('\t' + key + ' _gene_effect')
			if pd.isnull(self.sess.run(self._gene_effect_growth[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._gene_effect_growth[%s]" %key
			print('\t' + key + ' _selected_efficacies')
			if pd.isnull(self.sess.run(self._selected_efficacies[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._selected_efficacies[%s]" %key
			print('\t' + key + ' _out')
			if pd.isnull(self.sess.run(self._out[key], self.run_dict)).sum().sum() > 0:
				assert False, "nulls found in self._out[%s]" %key
			if (self.sess.run(self._out[key], self.run_dict) < 0).sum().sum() > 0:
				assert False, "negatives found in self._out[%s]" %key
			print('\t' + key + ' _output_norm')
			df = self.sess.run(self._output_norm[key], self.run_dict)
			if np.sum(pd.isnull(df).sum()) > 0:
				assert False, "%i out of %i possible nulls found in self._output_norm[%s]" % (
					np.sum(pd.isnull(df).sum()), np.prod(df.shape), key
					)
			if np.sum((df < 0).sum()) > 0:
				assert False, "negative values found in output_norm[%s]" % key
			print('\t' + key + ' _rpm')
			if np.sum(pd.isnull(self.sess.run(self._rpm[key], self.run_dict)).sum()) > 0:
				assert False, "nulls found in self._rpm[%s]" %key
			min_rpm = self.sess.run(self._rpm[key], self.run_dict).min().min()
			if min_rpm < 0:
				raise ValueError("Negative Reads Per Million (RPM) found (%f)" % min_rpm)

			min_output_norm = self.sess.run(self._output_norm[key], self.run_dict).min().min()
			if min_output_norm < 0:
				raise ValueError("Negative predicted normalized reads (output_norm) found (%f)" % min_output_norm)
			print('\t' + key + ' _cost_presum')
			df = self.cost_presum[key]
			print("sess run")
			if np.sum(pd.isnull(df).sum()) > 0:
				print(df)
				print()
				print(self.sess.run(
					 tf.log(1 + self._excess_variance_expanded[key] * 1e6 * self._output_norm[key]), self.run_dict)
				)
				print()
				print(self.sess.run(
					(self._rpm[key]+1e-6) * tf.log(self._excess_variance_expanded[key] * (self._rpm[key] + 1e-6) )
					, self.run_dict)
				)
				raise ValueError("%i nulls found in self._cost_presum[%s]" % (pd.isnull(df).sum().sum(), key))
			print('\t' + key + ' _cost')
			if pd.isnull(self.sess.run(self._cost, self.run_dict)):
				assert False, "Cost is null"
			print('\t' + key + ' _full_costs')
			if pd.isnull(self.sess.run(self._full_cost, self.run_dict)):
				assert False, "Full cost is null"



    ################################################################################################
	####################   T R A I N I N G    M  E  T  H  O  D  S    ###############################
	################################################################################################


	def step(self, ge_only=False):
		if ge_only:
			self.sess.run(self._ge_only_step, self.run_dict)
		else:
			self.sess.run(self._step, self.run_dict)
		self.epoch += 1


	def train(self, nepochs=800, starting_learn_rate=1e-4, burn_in_period=50, ge_only=100, report_freq=50,
			essential_genes=None, nonessential_genes=None, additional_metrics={}):
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
					print('%i epochs trained, time taken %s, projected remaining %s' % 
						(i+1, timedelta(seconds=round(delta)), timedelta(seconds=round(projected)))
					)
				print('cost', self.cost)
				#print('pDNA MSE', {key: self.sess.run(self._initial_cost[key], self.run_dict) for key in self.keys})
				print('relative_growth_rate')
				for key, val in self.growth_rate.items():
					print('\t%s max %1.3f, min %1.5f, mean %1.3f' % (
				    	key, val[val!=0].max(), val[val!=0].min(), val[val!=0].mean()))
				print('mean guide efficacy', self.guide_efficacy.mean())
				print('initial_offset SD: %r' % [(key, self.initial_offset[key].std()) for key in self.keys]) 
				print()
				ge = self.gene_effect
				print('gene mean', ge.mean().mean())
				print('SD of gene means', ge.mean().std())
				print("Mean of gene SDs", ge.std().mean())
				for key, val in additional_metrics.items():
					print(key, val(ge))
				if essential_genes is not None:
					print("Fraction Ess gene scores in bottom 15%%:", (ge.rank(axis=1, pct=True)[essential_genes] < .15).mean().mean()
					)
					print("Fraction Ess gene medians in bottom 15%%:", (ge.median().rank(pct=True)[essential_genes] < .15).mean()
					)
				if nonessential_genes is not None:
					print("Fraction Ness gene scores in top 85%%:", (ge.rank(axis=1, pct=True)[nonessential_genes] > .15).mean().mean()
					)
					print("Fraction Ness gene medians in top 85%%:", (ge.median().rank(pct=True)[nonessential_genes] > .15).mean()
					)

				print('\n\n')


	def save(self, directory, overwrite=False):
		if os.path.isdir(directory) and not overwrite:
			raise ValueError("Directory %r exists. To overwrite contents, use `overwrite=True`" % directory)
		elif not os.path.isdir(directory):
			os.mkdir(directory)

		write_hdf5(self.gene_effect, os.path.join(directory, "chronos_ge_unscaled.hdf5"))
		pd.DataFrame({"efficacy": self.guide_efficacy}).to_csv(os.path.join(directory,  "guide_efficacy.csv"))
		pd.DataFrame(self.cell_efficacy).to_csv(os.path.join(directory,  "cell_line_efficacy.csv"))
		pd.DataFrame(self.growth_rate).to_csv(os.path.join(directory,  "cell_line_growth_rate.csv"))
		pd.DataFrame({'screen_delay': self.screen_delay}).to_csv(os.path.join(directory,  "screen_delay.csv"))


	def snapshot(self):
		try:
			summary, cost = self.sess.run([self._merged, self._full_cost], self.run_dict)
			self.writer.add_summary(summary, self.epoch)
		except AttributeError:
			raise RuntimeError("missing writer for creating snapshot, probably because no log directory was supplied to Chronos")


	def __del__(self):
		for handle in self.persistent_handles:
			tf.delete_session_tensor(handle)


	################################################################################################
    ########################    A  T  T  R  I  B  U  T  E  S    ####################################
    ################################################################################################

	def inverse_efficacy(x):
		if not all((x <= 1) & (x > 0)):
			raise ValueError("efficacies must be greater than 0 and less than or equal to 1, received %r" % x)
		return -np.log(x)		
	
	@property
	def cost(self):
		return self.sess.run(self._cost, self.run_dict)

	@property
	def test_cost(self):
		return self.sess.run(self._test_cost, self.run_dict)

	@property
	def out(self):
		return {key: 
				pd.DataFrame(np.array(self.sess.run(v, self.run_dict)),
							index=self.index_map[key], columns=self.column_map[key])
			for key, v in self._out.items()}
	
	@property
	def output_norm(self):
		return {key: pd.DataFrame(self.sess.run(self._output_norm[key], self.run_dict), 
							  index=self.index_map[key], columns=self.column_map[key])
				for key in self.keys}
													
	@property
	def cost_presum(self):
		return {key: pd.DataFrame(self.sess.run(self._cost_presum[key], self.run_dict), 
							  index=self.index_map[key], columns=self.column_map[key])
				for key in self.keys}
	
	@property
	def guide_efficacy(self):
		return pd.Series(self.sess.run(self._guide_efficacy)[0][:-1], index=self.all_guides)
	@guide_efficacy.setter
	def guide_efficacy(self, desired_efficacy):
		self.sess.run(self.v_guide_efficacy.assign(
			Chronos.inverse_efficacy(np.array(list(desired_efficacy.loc[self.all_guides]) + [1e-16])).reshape((1, -1)) + 1e-16
		))

	@property
	def cell_efficacy(self):
		return {key: pd.Series(self.sess.run(self._cell_efficacy[key])[:, 0], index=self.all_cells).loc[self.cells[key]] for key in self.keys}
	@cell_efficacy.setter
	def cell_efficacy(self, desired_efficacy):
		for key in self.keys:
			missing = set(self.cells[key]) - set(desired_efficacy[key].index)
			if len(missing) > 0:
				raise ValueError("tried to assign cell efficacy for %s but missing %r" % (key, missing))
			try:
				self.sess.run(self.v_cell_efficacy[key].assign(
					Chronos.inverse_efficacy(desired_efficacy[key].loc[self.all_cells].fillna(1).values).reshape((-1, 1))
				))
			except ValueError as e:
				print(key)
				print(desired_efficacy[key].sort_values())
				raise e

	@property
	def growth_rate(self):
		return {key: pd.Series(self.sess.run(self._growth_rate[key])[:, 0], index=self.all_cells).loc[self.cells[key]]
				for key in self.keys}
	@growth_rate.setter
	def growth_rate(self, desired_growth_rate):
		for key, val in desired_growth_rate.items():
			missing = set(self.cells[key]) - set(desired_growth_rate[key].index)
			if len(missing) > 0:
				raise ValueError("tried to assign cell efficacy for %s but missing %r" % (key, missing))
			self.sess.run(self.v_growth_rate[key].assign(
				val.loc[self.all_cells].fillna(1).values.reshape((-1, 1)))
			)
	
	@property
	def gene_effect(self):
		mask = self.sess.run(self._gene_effect_mask)
		array = self.sess.run(self._combined_gene_effect)
		#array[mask == 0] = np.nan
		return Chronos.default_timepoint_scale * pd.DataFrame(array,
							 index=self.all_cells, columns=self.all_genes
							)
	@gene_effect.setter
	def gene_effect(self, desired_effect):
		mask = self.sess.run(self._gene_effect_mask)
		de = desired_effect.loc[self.all_cells, self.all_genes]
		if ((desired_effect.notnull() + mask) == 1).any().any():
			print("Warning: received some nonull values for genes in cell lines that have no guides targeting them, or inappropriate null values")
		de[mask == 0] = np.nan 
		means = de.mean().values.reshape((1, -1))
		residue = pd.DataFrame(de.values - means).fillna(0).values
		self.sess.run(self.v_mean_effect.assign(
			1.0/Chronos.default_timepoint_scale * means
			))
		self.sess.run(self.v_residue.assign(
			1.0/Chronos.default_timepoint_scale * residue
			))

	@property
	def screen_delay(self):
		return 1./Chronos.default_timepoint_scale * pd.Series(self.sess.run(self._screen_delay, self.run_dict)[0], index=self.all_genes)
	@screen_delay.setter
	def screen_delay(self, desired_screen_delay):
		self.sess.run(self._screen_delay.assign(Chronos.default_timepoint_scale * desired_screen_delay.loc[self.all_genes].values.reshape((1, -1))))

	@property
	def initial_offset(self):
		return {key:
			pd.Series(np.log(self.sess.run(self._initial_offset[key])[:, 0]),
				index=self.column_map[key]
				)
			for key in self.keys
			}

	@property
	def efficacy(self):
		return pd.DataFrame(self.sess.run(self._efficacy, self.run_dict),
					index=self.all_cells, columns=self.all_guides)
	

	@property
	def days(self):
		return {key: pd.Series(self.sess.run(self._days)[:, 0], index=self.index_map[key])
				for key in self.keys}

	@property
	def initial(self):
		return {key: pd.DataFrame(self.sess.run(self._initial[key], self.run_dict),
			index=self.index_map[key], columns=self.column_map[key]
			) for key in self.keys}


	@property
	def rpm(self):
		return {key: pd.DataFrame(self.sess.run(self._rpm[key], self.run_dict),
			index=self.index_map[key], columns=self.column_map[key]
			) for key in self.keys}

	@property
	def learning_rate(self):
		return self.run_dict[self._learning_rate]
	@learning_rate.setter
	def learning_rate(self, desired_learning_rate):
		self.run_dict[self._learning_rate] = desired_learning_rate

