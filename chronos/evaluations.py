
from warnings import warn
import numpy as np
import pandas as pd
from colorsys import hsv_to_rgb, rgb_to_hsv

try:
	from matplotlib import pyplot as plt
	from matplotlib.patches import Patch
	import seaborn as sns
	from scipy.stats import pearsonr
	from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
	from sklearn.decomposition import PCA
	from statsmodels.stats.multitest import fdrcorrection
except ModuleNotFoundError:
	raise ModuleNotFoundError("matplotlib, seaborn, statsmodels, scipy, and sklearn are required for the evaluations submodule. Try \
`pip install matplotlib; pip install seaborn; pip install scikit-learn; pip install statsmodels`")

from .model import powerset
from .plotting import density_scatter, lowess_trend, identify_outliers_by_zscore, identify_outliers_by_trend
from .plotting import binplot, dict_plot, identify_outliers_by_diagonal
try:
	from umap.umap_ import UMAP
	umap_present = True
except ModuleNotFoundError:
	warn("umap module not found. Some plots can't be made without it. Try `pip install umap-learn")
	umap_present = False
except NameError:
	warn("UMAP class not found where expected. Your umap module may be out of date. \
Try updating your version with `pip install --upgrade umap-learn")
	umap_present = False

try:
	from adjustText import adjust_text
	adjustText_present = True
except ModuleNotFoundError:
	warn("adjustText not found, which means labels in plots will not be adjusted to avoid overlap.\
Try `pip install adjustText`")



# UTILITIES


def np_cor_no_missing(x, y):
	"""Full column-wise Pearson correlations of two matrices with no missing values."""
	xv = (x - x.mean(axis=0))/x.std(axis=0)
	yv = (y - y.mean(axis=0))/y.std(axis=0)
	result = np.dot(xv.T, yv)/len(xv)
	return result


def group_cols_with_same_mask(x):
	"""
	Group columns with the same indexes of NAN values.
	
	Return a sequence of tuples (mask, columns) where columns are the column indices
	in x which all have the mask.
	"""
	per_mask = {}
	for i in range(x.shape[1]):
		o_mask = np.isfinite(x[:, i])
		o_mask_b = np.packbits(o_mask).tobytes()
		if o_mask_b not in per_mask:
			per_mask[o_mask_b] = [o_mask, []]
		per_mask[o_mask_b][1].append(i)
	return per_mask.values()


def fast_cor_core(x, y):
	'''
	x (`np.array`): 2D array. All columns will be correlated with all columns of y.
	y (`np.array`): 2D array. All columns will be correlated with all columns of x. 
					Must have save length as x.
	returns: `np.array` of shape (x.shape[1], y.shape[1]), where the ith, jth element
		is the pearson correlation of x[:, i] and y[:, j] with null elements removed.
	'''
	result = np.zeros(shape=(x.shape[1], y.shape[1]))

	x_groups = group_cols_with_same_mask(x)
	y_groups = group_cols_with_same_mask(y)
	for x_mask, x_columns in x_groups:
		for y_mask, y_columns in y_groups:
			# print(x_mask, x_columns, y_mask, y_columns)
			combined_mask = x_mask & y_mask

			# not sure if this is the fastest way to slice out the relevant subset
			x_without_holes = x[:, x_columns][combined_mask, :]
			y_without_holes = y[:, y_columns][combined_mask, :]

			try:
				c = np_cor_no_missing(x_without_holes, y_without_holes)
			except ValueError:
				raise ValueError("trying to correlate two groups with shapes %r and %r" %(
					x_without_holes.shape, y_without_holes.shape
				))
			# update result with these correlations
			result[np.ix_(x_columns, y_columns)] = c
	return result


def fast_cor(x, y=None):
	'''
	x (`pd.DataFrame`): Numerical matrix. All columns will be correlated with all columns of y.
	y (`pd.DataFrame`): Numerical matrix. All columns will be correlated with all columns of x. 
					Index must overlap x.
	returns: `pd.DataFrame` of shape (x.shape[1], y.shape[1]), where the ith, jth element
		is the pearson correlation of x[:, i] and y[:, j] with null elements removed.
	'''
	if y is None:
		y = x
	if x is y:
		shared = x.index
	else:
		shared = sorted(set(x.index) & set(y.index))
	if len(shared) < 2:
		raise ValueError("x and y don't have at least two rows in common")
	out = pd.DataFrame(fast_cor_core(x.loc[shared].values, y.loc[shared].values),
		index=x.columns, columns=y.columns)
	return out
	

def get_aligned_mutation_matrix(base_matrix, gene_effect):
	'''Aligning a mutation matrix with gene effect, requiring a minimum number of non-null values in gene effect'''
	aligned_matrix = base_matrix.reindex(gene_effect.index).fillna(False)
	aligned_matrix = aligned_matrix[sorted(set(aligned_matrix.columns) & set(gene_effect.columns))]
	aligned_matrix.fillna(False, inplace=True)
	aligned_matrix[gene_effect[aligned_matrix.columns].isnull()] = np.nan
	aligned_matrix = aligned_matrix[aligned_matrix.columns[
		(aligned_matrix & gene_effect[aligned_matrix.columns].notnull() ).sum() > 2
	]]
	return aligned_matrix


def split_color(rgb):
	''' get two colors with the same hue and saturation but different values'''
	h, s, v = rgb_to_hsv(*rgb)
	return hsv_to_rgb(h, s, .3), hsv_to_rgb(h, s, .6)


def generate_powerset_palette(keys, start='random',
							 base_saturation=1, base_hsv_value=.7):
	'''
	Generate a palette for the powerset of `keys`. Colors for the individual keys will be evenly spaced
	in hue space. Combinations will have the average of the hues of each key, with identical hues being resolved
	by different hsv values (brightness).
	Parameters:
		`keys` (iterable): the base keys that will be combined into a powerset
		`start` (`float` or "random"): optional hue for the first entry in `keys`
		`base_saturation`: saturation of colors for the individual keys
		`base_hsv_value`: hsv value parameter for colors for the individual keys
	Returns:
		`dict` with an entry for each possible unique combination of the keys (excluding the empty set) containing
			an RGB color for the combination
	'''
	if start == 'random':
		start = np.random.uniform()
	base_hues = start + np.arange(len(keys))/len(keys)
	base_rgb = dict(zip(keys, [hsv_to_rgb(hue, base_saturation, base_hsv_value) for hue in base_hues]))
	out = {}
	keysets = list(powerset(keys))
	for keyset in keysets:
		if not len(keyset):
			continue
		color = np.mean(np.stack([base_rgb[key] for key in keyset]), axis=0)
		out[keyset] = tuple(color)
	for i, keyset1 in enumerate(keysets):
		if not len(keyset1):
			continue
		for keyset2 in keysets[i+1:]:
			if not len(keyset2):
				continue
			dist = np.sqrt(((np.array(out[keyset1]) - np.array(out[keyset2]))**2).sum())
			if dist < .1:
				out[keyset1], out[keyset2] = split_color(out[keyset1])
	return out


def trim_overlapping_lead_and_tail(strings):
	'''
	Removes extraneous prefixes/suffixes common to all the strings for more parsimonious labeling
	'''
	if len(strings) < 2:
		return strings
	n = min([len(string) for string in strings])
	for i in range(n):
		c = strings[0][i]
		if any([string[i] != c for string in strings[1:]]):
			break
	if i == n:
		raise ValueError("Shortest string has no distinct substring:\n%r" % strings)
	for j in range(n):
		c = strings[0][-j-1]
		if any([string[-j-1] != c for string in strings[1:]]):
			break
	if j == 0:
		return [string[i:] for string in strings]
	return [string[i:-j] for string in strings]


def _strip_identical_prefix(s1, s2):
	i = 0
	while s1[i] == s2[i]:
		i += 1
	return s1[i:], s2[i:]

def _make_aliases(keys):
	'''
	Tries to create a series with values holding a unique, logical two-letter code
	for each key in keys.
	'''
	deduplicated = pd.Series([s for s in trim_overlapping_lead_and_tail(keys)], index=[s for s in keys])
	true_unique = deduplicated.copy()
	for i in range(len(deduplicated)):
		for j in range(i+1, len(deduplicated)):
			true_unique.iloc[i], true_unique.iloc[j] = _strip_identical_prefix(true_unique[i], true_unique[j])
	out = {}
	for s in keys:
		if deduplicated[s].startswith(true_unique[s]):
			out[s] = deduplicated[s][:2]
		else:
			out[s] = deduplicated[s][0] + true_unique[s][0]
	print(deduplicated, true_unique)
	return pd.Series(out)


def append_to_legend_handles(lines, ax):
	'''
	Add text to the matplotlib legend to an axis
	Parameters:
		`lines` (iterable of `str`): lines to add
		`ax`: `matplotlib.Axis`
	Returns:
		legend handles
	'''
	handles, labels = ax.get_legend_handles_labels()
	for line in lines:
		handles.append(
			Patch( 
				color=(0, 0, 0, 0), 
				label=line 
			)
		)
	return handles


# METRICS

def mad(x, axis=None):
	'''median absolute deviation from the median'''
	x = x[pd.notnull(x)]
	med = np.median(x, axis)
	return np.median( np.abs(x-med), axis)


def nnmd(pos, neg):
	'''null-normalixed median difference between the `pos` and `neg` arrays'''
	return (np.median(pos[pd.notnull(pos)]) - np.median(neg[pd.notnull(neg)]))/mad(neg)


def auroc(pos, neg):
	'''ROC AUC of separation between `pos` and `neg` arrays'''
	pos = pos[pd.notnull(pos)]
	neg = neg[pd.notnull(neg)]
	true = [0] * len(pos) + [1] * len(neg)
	return roc_auc_score(y_true=true, y_score=list(pos) + list(neg))

def pr_auc(pos, neg):
	'''Area under precision-recall curve separating `pos` and `neg` arrays'''
	pos = pos[pd.notnull(pos)]
	neg = neg[pd.notnull(neg)]
	probas = np.concatenate([np.array(pos), np.array(neg)])
	true = [0] * len(pos) + [1] * len(neg)
	precision, recall, thresh = precision_recall_curve(y_true=true, probas_pred=probas)
	return auc(recall, precision)


# PRE RUN PLOTS

def replicate_plot(readcounts, rep1, rep2):
	'''
	Given a `pandas.DataFrame` matrix of `readcounts` with replicates as rows and sgRNAs as columns, plot
	the logged readcounts of `rep1` vs `rep2` and annotate with their Pearson correlation.
	'''
	for rep in rep1, rep2:
		if not rep in readcounts.index:
			raise ValueError("replicate label %s not found in the index of `readcounts`" % rep)
	x = np.log2(readcounts.loc[rep1]+1)
	y = np.log2(readcounts.loc[rep2]+1)
	density_scatter(x, y,
							label_outliers=2)
	plt.xlabel("%s Readcounts (+1Log)" % rep1)
	plt.ylabel("%s Readcounts (+1Log)" % rep2)
	r = x.corr(y)
	plt.text(s='R = %1.2f' % r, x=.05, y=.9, transform=plt.gca().transAxes)


def all_replicate_plot(readcounts, sequence_map, cell_line, plot_width):
	'''
	Given a `pandas.DataFrame` matrix of `readcounts` with replicates as rows and sgRNAs as columns, generate a
	`replicate_plot` for all pairs of replicates of `cell_line`. See `chronos.Chronos` for a description
	of `sequence_map`. `plot_width` gives the plot width in inches.
	'''
	reps = sequence_map.query("cell_line_name == %r" % cell_line).sequence_ID.unique()
	rep_labels = dict(zip(reps, trim_overlapping_lead_and_tail(reps)))
	n = 0
	titles = {}
	for i in range(len(reps)-1):
		for j in range(i+1, len(reps)):
			n += 1
			titles["%s %i" % (cell_line, n)] = (reps[i], reps[j])
	def plotfunc(x):
		replicate_plot(readcounts, *x)
		plt.xlabel("Rep." + rep_labels[x[0]])
		plt.ylabel("Rep." + rep_labels[x[1]])
	dict_plot(titles, plotfunc, plot_width)
	plt.tight_layout()


def pDNA_plot(readcounts, sequence_map, rep, sgrnas=None):
	'''
	Given a `pandas.DataFrame` matrix of `readcounts` with replicates as rows and sgRNAs as columns, plot
	the logged readcounts of `rep` vs median readcounts in pDNA of the same batch. `sgrnas` optionally
	subsets the plot to the specified sgrnas. See `chronos.Chronos` for a description
	of `sequence_map`.
	'''
	if not rep in readcounts.index:
		raise ValueError("Rep %r not in readcounts index" %rep)
	if not sgrnas is None:
		controls = sorted(set(sgrnas) & set(readcounts.columns))
		if not controls:
			raise ValueError("None of the specified sgRNAs are in the readcounts columns: \n%r" 
							% sgrnas)
	batch_label = sequence_map.query("sequence_ID == %r" % rep).iloc[0]['pDNA_batch']
	pdna_seq = sequence_map\
				.query("cell_line_name == 'pDNA'")\
				.query("pDNA_batch == %r" % batch_label)\
				.sequence_ID
	pdna = np.log2(readcounts.loc[pdna_seq]+1).median()
	ltp = np.log2(readcounts.loc[rep]+1)
	if not sgrnas is None:
		pdna = pdna.loc[controls]
		ltp = ltp.loc[controls]
	density_scatter(pdna, ltp,
							trend_line=False, diagonal=True)
	plt.xlabel("%s Readcounts (+1Log)" % "pDNA")
	plt.ylabel("%s Readcounts (+1Log)" % rep)
	r = ltp.corr(pdna)
	plt.text(s='R = %1.2f' % r, x=.05, y=.9, transform=plt.gca().transAxes)



def paired_pDNA_plots(readcounts, sequence_map, cell_line, 
					  negative_control_sgRNAs=None, positive_control_sgRNAs=None,
					 plot_width=7.5, plot_height=3, page_height=9):
	'''
	If `negative_control_sgRNAs` and `positive_control_sgRNAs` is none,
	produces one subplot for each replicate of the cell line with a `pDNA_plot`.
	Otherwise, generates pairs of pDNA plots for each replicate. If both control types
	are supplied, one will be plotted on each side. If one is missing,
	it will be replaced with all sgRNAs. `plot_width` specified the figure width in inches,
	but `plot_height` specifies subplot height. This will be adjusted if the total figure
	height would exceed `page_height`. See `pDNA_plot` for other parameters.
	'''
	reps = sequence_map.query("cell_line_name == %r" % cell_line).sequence_ID.unique()
	labels = dict(zip(reps, trim_overlapping_lead_and_tail(reps)))
	left_title = "Negative Controls"
	right_title = "Positive Controls"
	
	if negative_control_sgRNAs is None and positive_control_sgRNAs is None:
		titles = dict(zip(trim_overlapping_lead_and_tail(reps), reps))
		def plotfunc(x):
			pDNA_plot(readcounts, sequence_map, x)
			plt.ylabel(labels[x])
		dict_plot(titles, plotfunc)
		return
	elif positive_control_sgRNAs is None:
		positive_control_sgRNAs = readcounts.columns
		right_title = "All sgRNAs"
	elif negative_control_sgRNAs is None:
		negative_control_sgRNAs = readcounts.columns
		left_title = "All sgRNAs"
		
	height = min(page_height, plot_height*len(reps))
	fig, axes = plt.subplots(len(reps), 2, figsize=(plot_width, height))
	for i, rep in enumerate(reps):
		plt.sca(axes[i, 0])
		pDNA_plot(readcounts, sequence_map, rep, negative_control_sgRNAs)
		plt.ylabel('Rep. ' + labels[rep])
		plt.title(left_title)
		
		plt.sca(axes[i, 1])
		pDNA_plot(readcounts, sequence_map, rep, positive_control_sgRNAs)
		plt.ylabel("")
		plt.title(right_title)



# POST RUN PLOTS


def gene_outlier_plot(gene_effect1, gene_effect2,
	xlabel="gene_effect1 zscore", ylabel="gene_effect2 zscore",
		ax=None, legend=True,
		density_scatter_args={"label_outliers": 10, "trend_line": True}, 
		legend_args={}, metrics=None
):
	'''
	Compares the most extreme outliers for the matrices `gene_effect1` and `gene_effect2` 
	with a density scatter of the maximum and minimum screen gene effect for each gene
	with results from `gene_effect1` on one axis and `gene_effect2` on the other, and returns the 
	outliers from the trend line.
	This plot is useful to detect if one method is producing etreme outliers within a gene score
	relative to the other.
	'''
	gene_effect1, gene_effect2 = gene_effect1.align(gene_effect2, join="inner")
	zscore1 = (gene_effect1 - gene_effect1.mean())/gene_effect1.std()
	zscore2 = (gene_effect2 - gene_effect2.mean())/gene_effect2.std()
	mins1 = zscore1.min()
	mins1.index = [s + "_Min" for s in mins1.index]
	max1 = zscore1.max()
	max1.index = mins1.index
	mins2 = zscore2.min()
	mins2.index = [s + "_Min" for s in mins2.index]
	max2 = zscore2.max()
	max2.index = mins2.index
	x = pd.concat([mins1, max1])
	y = pd.concat([mins2, max2])
	if not ax:
		ax = plt.gca()
	plt.sca(ax)
	density_scatter(x, y, **density_scatter_args)
	plt.title("Most Extreme Values by ZScore")
	out = {
		'low_outliers': mins1.index[identify_outliers_by_trend(mins1, mins2, 5)],
		'high_outliers': max1.index[identify_outliers_by_trend(max1, max2, 5)]
	}
	if metrics is None:
		return out
	else:
		metrics.update(out)




def gene_corr_vs_mean(gene_effect1, gene_effect2, ax=None,
					  legend=True,
					 density_scatter_args={"label_outliers": 5, "trend_line": False}, legend_args={}, metrics=None
					 ):
	'''
	Shows the correlation of a gene's gene effect profile within the two matrices `gene_effect` and `gene_effect2`
	with the gene's mean effect (averaged between the two matrices) on the x axis, and returns the genes with lowest correlation.
	'''
	corrs = gene_effect1.corrwith(gene_effect2).dropna()
	means = .5*(gene_effect1[corrs.index].mean() + gene_effect2[corrs.index].mean())
	density_scatter(means, corrs, **density_scatter_args)
	if not ax:
		ax = plt.gca()
	ax.set_xlabel("Gene Mean")
	ax.set_ylabel("Gene Correlation")

	out = {
		"gene_corr_med": corrs.median(),
		"gene_corr_lt_9": (corrs < .9).sum(),
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.3f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	out["lowest_corr"] = corrs.sort_values().index[0:20]
	if metrics is None:
		return out
	else:
		metrics.update(out)


def gene_corr_vs_mean_diff(gene_effect1, gene_effect2, ax=None,
					  legend=True,
					 density_scatter_args={"label_outliers": 5, "trend_line": False, "outliers_from": "xy_zscore"},
					  legend_args={}, metrics=None
					 ):
	'''
	Shows the correlation of a gene's gene effect profile within the two matrices `gene_effect` and `gene_effect2`
	with the difference in the gene's mean effect between the two matrices on the x axis. This plot is useful for
	seeing which genes have the most disagreement between the matrices either by correlation or by mean effect.
	It returns outliers found by zscore, i.e. genes with lowest agreement taking into account both their means
	and their correlations.
	'''
	corrs = gene_effect1.corrwith(gene_effect2).dropna()
	mean_diff = gene_effect1[corrs.index].mean() - gene_effect2[corrs.index].mean()
	corrs, mean_diff =  corrs.align(mean_diff.dropna(), join="inner")
	density_scatter(mean_diff, corrs, **density_scatter_args)
	if not ax:
		ax = plt.gca()
	ax.set_xlabel("Gene Mean Diff")
	ax.set_ylabel("Gene Correlation")

	out = {
		"gene_corr_med": corrs.median(),
		"gene_corr_lt_9": (corrs < .9).sum(),
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.3f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	out["worst_agreement"] = corrs.index[identify_outliers_by_zscore(corrs, mean_diff, 10)]
	if metrics is None:
		return out
	else:
		metrics.update(out)


def control_histogram(gene_effect, positive_control_genes, negative_control_genes, ax=None,
					  legend=True,
					 kde_args={}, legend_args={}, metrics=None):
	'''
	Produces KDE plots of the distribution of positive and negative control gene scores 
	in the gene effect matrix. Both the mean gene scores and the raveled gene scores are
	shown. Control separatioon results measured by NNMD and AUROC are returned.
	'''
	pos = gene_effect.reindex(columns=positive_control_genes).dropna(axis=1, how='all')
	neg = gene_effect.reindex(columns=negative_control_genes).dropna(axis=1, how='all')
	sns.kdeplot(pos.mean(), bw_adjust=.5, fill=True, alpha=.3, lw=0, color="red", 
				label="Positive Control Means", ax=ax, gridsize=1000, **kde_args)
	sns.kdeplot(neg.mean(), bw_adjust=.5, fill=True, alpha=.3, lw=0, color="blue",
				label="Negative Control Means", ax=ax, gridsize=1000, **kde_args)
	sns.kdeplot(pos.stack(), bw_adjust=.5, lw=2, color="crimson", 
				label="Positive Control Scores", ax=ax, gridsize=1000, **kde_args)
	sns.kdeplot(neg.stack(), bw_adjust=.5, lw=2, color="navy",
				label="Negative Control Scores", ax=ax, gridsize=1000, **kde_args)
	if not ax:
		ax = plt.gca()
	ax.set_xlabel("Gene Effect")
	out = {
		"NNMD_of_means": nnmd(pos.mean(), neg.mean()),
		"NNMD_of_scores": nnmd(pos.stack(), neg.stack()),
		"AUROC_of_means": auroc(pos.mean(), neg.mean()),
		"AUROC_of_scores": auroc(pos.stack(), neg.stack())
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.3f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def mean_vs_sd_scatter(gene_effect,
						   ax=None,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={"alpha": .6, "s": 10, "label_outliers": 3, "outliers_from": "xy_zscore"}
						  ):
	'''
	Plots the gene mean vs its standard deviation for each gene in the matrix `gene_effect`.
	'''
	means = gene_effect.mean()
	sd = gene_effect.std()/means.std()
	
	if not ax:
		ax = plt.gca()
	density_scatter(means, sd, ax=ax, **density_scatter_args)
	plt.ylabel("Gene SD / SD of Gene Means")
	plt.xlabel("Gene Mean")
	
	out = {
		"mean_SD:SD_means": sd.mean()
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def mean_vs_cell_eff_correlation(gene_effect, cell_efficacy,
		ax=None, metrics=None, legend=True, legend_args={"loc": "upper right"},
		density_scatter_args={"alpha": .6, "s": 10, "label_outliers": 5, "outliers_from": "y"}
	):
	'''
	Usiung the matrix `gene_effect`, plots each gene's gene effect profile's mean vs its correlation with estimated
	`cell_efficacy` (`pandas.Series` or `numpy.ndarray`) of all screens.
	This plot is useful for detecting screen quality bias, in that genes with lower means will tend to 
	be negatively correlated with cell efficacy. Returns cell efficacy correlation mean, standard deviation,
	and it correlation with gene mean.
	'''
	means = gene_effect.mean()
	corrs = gene_effect.corrwith(cell_efficacy)
	if not ax:
		ax = plt.gca()
	density_scatter(means, corrs, ax=ax, **density_scatter_args)
	plt.ylabel("Gene Effect R with Cell Efficacy")
	plt.xlabel("Gene Mean")
	out = {
		"cell_efficacy_corr_mean": corrs.mean(),
		"cell_efficacy_corr_sd": corrs.std(),
		"cell_efficacy_corr_gene_mean_trend": means.corr(corrs)
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def screen_nnmd_auroc_scatter(gene_effect, positive_control_genes, negative_control_genes, ax=None,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={}):
	'''
	For each screen (row) in the matrix `gene_effect`, computes the separation of the iterable `positive_control_genes`
	from `negative_control_genes` by NNMD and AUROC. This is useful for visualizing the distribution of screen quality.
	The median and mean of both measures are returned.
	'''
	poscon = sorted(set(positive_control_genes) & set(gene_effect.columns))
	negcon = sorted(set(negative_control_genes) & set(gene_effect.columns))
	nnmds = gene_effect.apply(
		lambda x: nnmd(x[poscon], x[negcon]),
		axis=1
			)
	aurocs = gene_effect.apply(
		lambda x: auroc(x[poscon], x[negcon]),
		axis=1
			)
	
	if not ax:
		ax=plt.gca()
	density_scatter(aurocs, nnmds, ax=ax, **density_scatter_args)
	plt.xlabel("AUROC - Higher is Better")
	plt.ylabel("NNMD - Lower is Better")
	
	out = {
		"NNMD_median": nnmds.median(),
		"NNMD_mean": nnmds.mean(),
		"AUROC_median": aurocs.median(),
		"AUROC_mean": aurocs.mean()
	}

	if legend:
		handles = append_to_legend_handles([
			"%s: %1.3f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="lower left", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out) 


def expression_addiction_volcano(gene_effect, addiction_expressions,
	max_threshold=-.2,
									 ax=None,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={"trend_line": False}
	):
	'''
	Given the matrix of expression `addiction_expressions`, whose columns should only include
	genes expected to be expression addictions (i.e. cause loss of viability in cell lines
	that overexpress them), computes the Pearson correlation and associated false discovery rate between
	the gene's expression and its gene effect in the matrix `gene_effect` and plots the result
	as a volcano. Note that the p-values informing the FDRs (q values) are optimistic due to
	the assumption of normal errors. The fraction of selective dependencies with FDR < 0.1 or
	R < -0.2 is returned.
	This plot is useful for evaluating the ability to identify 
	selective dependencies and their association with the correct biomarker.
	'''
	gene_effect, addiction_expressions = gene_effect.align(addiction_expressions, join="inner")
	corr = {}
	p = {}
	for gene in gene_effect:
		mask = gene_effect[gene].notnull() & addiction_expressions[gene].notnull()
		corr[gene], p[gene] = pearsonr(gene_effect[gene][mask], addiction_expressions[gene][mask])
	corr, p = pd.Series(corr), pd.Series(p)
	p /= 2
	p[corr > 0] = 1 - p[corr > 0]
	q = pd.Series(fdrcorrection(p.values, .05)[1], index=p.index)
	if ax is None:
		ax = plt.gca()
	plt.sca(ax)
	density_scatter(corr, -np.log10(q), **density_scatter_args)
	plt.xlabel("Expression/GE Correlation")
	plt.ylabel("-log10(FDR)")
	out = {
		"expression_addictions_FDR_0.10": (q < .1).mean(),
		 "expression_addictions_<_-0.2": (corr < -.2).mean()
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="upper right", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def selective_mutated_vs_not_scatter(gene_effect, mutation_matrix,
									 ax=None,
							 metrics=None, legend=True, legend_args={}, label_outliers=3,
							   scatter_args={"alpha": .75, "linewidth": 1, "cmap": 'viridis_r'}
	):
	'''
	A common pattern of dependency in cancer is "oncogene addiction," in which cells
	selectively require proteins with oncogenic gain of function mutations to maintain viability.
	Canonical examples include BRAF, NRAS, KRAS, CTNNB1, and EGFR. We expect cells with the 
	gain of function alteration to show more negative gene effect than those without. This
	function takes in a boolean `mutation_matrix` with gene columns and cell line rows,
	which should only include genes that have known oncogenic gain of function alterations and should
	be `True` for cell lines that have one of the known gain of function alterations.
	For each gene in `mutation_matrix`, its mean in `gene_effect` for cell lines without 
	gain of function is plotted on the x axis and its mean in cell lines with gain of function
	is plotted on the y axis. The separation of these two by NNMD and AUROC is returned 
	both as a median over genes of results per gene and by combining all GoF scores for all genes as positive controls
	and the same genes' scores in lines without their indicated GoF alterations as the negative
	controls ("total" NNMD/AUROC). This plot is useful for evaluating the ability to detect 
	selective dependencies and their biomarkers.
	'''
	gene_effect = gene_effect.dropna(how='any', axis=1)
	mutation_matrix = get_aligned_mutation_matrix(mutation_matrix, gene_effect)
	gene_effect, mutation_matrix = gene_effect.align(mutation_matrix, join="inner")
	mutation_matrix = mutation_matrix.dropna(how='all', axis=1)
	gene_effect = gene_effect.dropna(how='all', axis=1)
	gene_effect, mutation_matrix = gene_effect.align(mutation_matrix, join="inner")
	if gene_effect.shape[0] == 0 or gene_effect.shape[1] == 0:
		raise ValueError("Gene_effect and mutation_matrix have an axis with no overlaps (either genes or screens)")
	scale = np.log2(mutation_matrix.sum().astype(float))
	nnmds = mutation_matrix.apply(lambda x: nnmd(gene_effect[x.name][x], gene_effect[x.name][~x]),
								 axis=0)
	aurocs = mutation_matrix.apply(lambda x: auroc(gene_effect[x.name][x], gene_effect[x.name][~x]),
								 axis=0)
	total_nnmd = nnmd(gene_effect[mutation_matrix.fillna(False)].stack(), gene_effect[~mutation_matrix.fillna(False)].stack())
	total_auroc = auroc(gene_effect[mutation_matrix].stack(), gene_effect[~mutation_matrix].stack())
	pos_means = gene_effect[mutation_matrix].mean()
	neg_means = gene_effect[~mutation_matrix].mean()
	pos_means, neg_means = pos_means.align(neg_means, join="inner")
	
	if ax:
		plt.sca(ax) #needed because scatter doesn't accept ax arg?
	plt.scatter(neg_means, pos_means, s=10*scale, c=scale, **scatter_args)
	if label_outliers:
		outliers = identify_outliers_by_diagonal(neg_means, pos_means, label_outliers)
		texts = [plt.text(s=neg_means.index[i],x=neg_means[i], y=pos_means[i], fontsize=6, color=[.8, .3, .05]) for i in outliers]
		if adjustText_present:
			adjust_text(texts, x=neg_means.values, y=pos_means.values, arrowprops=dict(lw=1, arrowstyle="-", color="black"),
				expand_points=(2, 2.5))
	xlim = plt.gca().get_xlim()
	ylim = plt.gca().get_ylim()
	plt.plot(
		[min(xlim[0], ylim[0]), max(xlim[1], ylim[1])],
		[min(xlim[0], ylim[0]), max(xlim[1], ylim[1])],
		'--', color='tomato', lw=1
		)
	plt.colorbar(label="Log2(# Mutated Lines)")
	plt.xlabel("Gene Mean Without Mutation")
	plt.ylabel("Gene Mean With Mutation")
	
	out = {
		"selective_NNMD_gene_median": nnmds.median(),
		"selective_NNMD_raveled": total_nnmd,
		"selective_AUROC_gene_median": aurocs.median(),
		"selective_AUROC_raveled": total_auroc
	}
	if ax is None:
		ax = plt.gca()
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="upper left", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def copy_number_trend(gene_effect, copy_number,
	downsample=False, downsample_lower_quantile_bound=.05, downsample_upper_quantile_bound=.95,
								ax=None,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={"alpha": .75}
	):
	'''
	Produces a scatter of the raveled `gene_effect` matrix (y) vs the raveled `copy_number` matrix
	(x). USeful for visualizing how much depletion highly amplified regions can produce. 
	If `downsample` is a float between 0 and 1, points with CN between `downsample_lower_quantile` and 
	`downsample_upper_quantile` will be randomly reduced to the fraction given by `downsample`. 
	This can greatly increase plotting speed by reducing the number of uninformative plots
	with euploid CN being plotted.
	The overall correlation of the raveled gene effect and CN matrices is returned.
	'''
	gene_effect, copy_number = gene_effect.align(copy_number, join='inner')
	ge_raveled, cn_raveled = np.ravel(gene_effect), np.ravel(copy_number)
	mask = pd.notnull(cn_raveled) & pd.notnull(ge_raveled)
	out = {
		"raveled_CN_corr":  pearsonr(ge_raveled[mask], cn_raveled[mask])[0]
	}
	if downsample:
		ind = np.arange(len(cn_raveled))
		selection = np.random.binomial(p=downsample, n=1, size=len(cn_raveled))
		low = np.quantile(cn_raveled, downsample_lower_quantile_bound)
		high = np.quantile(cn_raveled, downsample_upper_quantile_bound)
		selection[cn_raveled < low] = 1
		selection[cn_raveled > high] = 1
		cn_raveled[selection == 0] = np.nan
		mask = pd.notnull(cn_raveled) & pd.notnull(ge_raveled)
	if ax is None:
		ax = plt.gca()
	plt.sca(ax)
	density_scatter(cn_raveled[mask], ge_raveled[mask], **density_scatter_args)
	plt.xlabel("Copy Number")
	plt.ylabel("Gene Effect")
	out = {
		"raveled_CN_corr":  pearsonr(ge_raveled[mask], cn_raveled[mask])[0]
	}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="upper right", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def copy_number_gene_corrs(gene_effect, copy_number,
								ax=None,
							 metrics=None, legend=True, legend_args={},
							   binplot_args={}
	):
		'''
		Computes the correlation for each gene in the matrix `gene_effect` with its copy nunber in the matrix
		`copy_number` (genes are columns, cell lines are rows), then bins by mean gene effect and plots the result.
		This is useful for idenfying the two types of copy number effect: double strand break toxicity,
		which causes nonessential genes to have gene effect negatively correlated with their own CN,
		and the copy buffering effect, which causes common essential genes to be positively correlated.
		'''
		gene_effect, copy_number = gene_effect.align(copy_number, join='inner')
		corrs = gene_effect.corrwith(copy_number)
		means = gene_effect.mean()
		if ax is None:
			ax = plt.gca()
		plt.sca(ax)
		binplot(means, corrs, **binplot_args)
		plt.xlabel("Gene Effect Mean")
		plt.ylabel("Gene Effect R with CN")
		return {}


def guide_estimate_corr_vs_sd_scatter(predicted_lfc, observed_lfc,
							  ax=None,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={}
							 ):
	'''
	Given the two matrices of log fold-change, one predicted by the model (`predicted_lfc`)
	with guides as columns and replicates as rows, computes the correlation between 
	each sgRNA and produces a scatter with the standard deviation of the sgRNAs in the observed
	matrix on the x axis. This is useful to see the agreement. In general sgRNAs with lower SD
	may have lower correlation as there is less signal. The median correlation of all sgRNAs
	and the median correlation of sgRNAs in the top 20% highest SD are returned. 
	'''
	corrs = predicted_lfc.corrwith(observed_lfc).dropna()
	sd = observed_lfc.std().loc[corrs.index]
	density_scatter(sd, corrs, ax=ax, **density_scatter_args)
	plt.xlabel("Guide LFC SD")
	plt.ylabel("Guide LFC Estimated/Observed R")
	
	out = {
		"corrs_median": corrs.median(),
		"corrs_median_20ile_most_variable": corrs[sd > sd.quantile(.8)].median()
	}
	if ax is None:
		ax = plt.gca()
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="lower right", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def guide_estimate_corr_vs_guide_efficacy_scatter(predicted_lfc, observed_lfc,
											 guide_efficacy,
							  ax=None,
							 metrics=None, legend=False, legend_args={},
							   density_scatter_args={}
							 ):
	'''
	Given the two matrices of log fold-change, one predicted by the model (`predicted_lfc`)
	with guides as columns and replicates as rows, computes the correlation between 
	each sgRNA, then plots that correlation with the `pandas.Series` `guide_efficacy`
	which should be estimated by the model. In general we expect lower fidelity between
	predicted and observed sgRNAs for guides with low efficacy. 
	'''
	corrs = predicted_lfc.corrwith(observed_lfc)
	guide_efficacy = guide_efficacy.loc[corrs.index]
	density_scatter(guide_efficacy, corrs, ax=ax, **density_scatter_args)
	plt.xlabel("Guide Efficacy")
	plt.ylabel("Guide LFC R")
	
	out = {
	}
	if ax is None:
		ax = plt.gca()
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="lower right", **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)  


def predicted_vs_observed_readcounts(predicted_readcounts, observed_readcounts,
							  ax=None, max_points=10000,
							 metrics=None, legend=True, legend_args={},
							   density_scatter_args={"alpha": .5, "s": 10, "diagonal":True}
							 ):
	'''
	Given the two normalized matrices of readcounts, one predicted by the model (`predicted_readcounts`)
	and one observed (`observed_readcounts`), 
	with guides as columns and replicates as rows, produces a scatter plot with observations
	on the x axis and predictions on y. Points are subsampled to `max_points`.
	For Chronos, very low observed readcounts will be systematically
	predicted to have more, due to the structure of counts noise (it is more likely to observe few counts
	if the real expectation is high than vice versa). If the total trend of readcounts is above or
	below the diagonal however, that may indicate a normalization problem with the normalization.
	Returns the correlation, mean difference (should be near 0), and median difference (should
	also be near 0)
	'''
	estimated = pd.DataFrame(np.log10(predicted_readcounts.values+1), 
								 index=predicted_readcounts.index, 
								 columns=predicted_readcounts.columns
								)
	observed = pd.DataFrame(np.log10(observed_readcounts.values+1), 
								 index=observed_readcounts.index, 
								 columns=observed_readcounts.columns
								)
	estimated, observed = estimated.align(observed, join="inner")
	stacked_est = np.ravel(estimated.values)
	stacked_obs = np.ravel(observed.values)
	
	if len(stacked_est) > max_points:
		chosen = np.random.choice(range(len(stacked_est)), size=max_points)
	else:
		chosen = range(len(stacked_est))
	stacked_obs = stacked_obs[chosen]
	stacked_est = stacked_est[chosen]
	mask = pd.notnull(stacked_est) & pd.notnull(stacked_obs)
	stacked_est = pd.Series(stacked_est[mask])
	stacked_obs = pd.Series(stacked_obs[mask])
	density_scatter(stacked_obs, stacked_est,
					ax=ax, **density_scatter_args)
	plt.xlabel("Observed Readcounts (Log10)")
	plt.ylabel("Estimated Readcounts (Log10)")
	diff = pd.DataFrame(observed.values-estimated.values,
		index=observed.index, columns=observed.columns)
	out = {
		"readcount_estimate_corr": pearsonr(stacked_est.values, stacked_obs.values)[0],
		"readcount_estimate_mean_displacement": diff.mean().mean(),
		"readcount_estimate_median_displacement": diff.median().median()
			}
	if ax is None:
		ax = plt.gca()
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, loc="lower right", fontsize=8, **legend_args)
	if metrics is None:
		return out
	else:
		metrics.update(out)


def lfc_corr_vs_excess_variance(predicted_lfc, observed_lfc, excess_variance,
							  ax=None,
							 metrics=None, legend=True, legend_args={'loc': 'upper right'},
							   density_scatter_args={"alpha": .5, "s": 10, 'trend_line': False, 'label_outliers':5}
							 ):
	'''
	Given the two matrices of log fold-change, one predicted by the model (`predicted_lfc`)
	with guides as columns and replicates as rows, computes the correlation between 
	each replicate, then correlates that correlation with the `pandas.Series` `guide_efficacy`
	which should be estimated by the model. In general we expect lower fidelity between
	predicted and observed sgRNAs for guides with low efficacy.
	'''
	corrs = predicted_lfc.corrwith(observed_lfc, axis=1).dropna()
	corrs, excess_variance = corrs.align(excess_variance.dropna(), join='inner')
	if ax is None:
		ax = plt.gca()
	if excess_variance.dropna().nunique() == 1:
		excess_variance = excess_variance + np.random.uniform(0, excess_variance.dropna().iloc[0]/10, size=len(excess_variance))
		ax.set_xlabel("Screen Excess Variance (Log10) jittered")
	else:
		ax.set_xlabel("Screen Excess Variance (Log10)")
	density_scatter(np.log10(excess_variance), corrs,
					ax=ax, **density_scatter_args)

	ax.set_ylabel("Correlation Predicted/Observed LFC")
	
	out = {
		"lfc_cell_corrs_median": corrs.median(),
		"lfc_cell_corrs_min": corrs.min(),
			}
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
	out['lfc_cell_corrs_low'] = corrs.sort_values().index[:10]
	if metrics is None:
		return out
	else:
		metrics.update(out)   



def _mean_ge_deviation_vs_grad(
		predicted_readcounts, observed_readcounts, guide_map, ge_mean_grad, 
			ax=None,
		metrics=None, legend=True, legend_args={},
		density_scatter_args={"alpha": .5, "s": 10}
):
	'''
	A Chronos-specific plot.
	Given the two normalized matrices of readcounts, one predicted by the model (`predicted_readcounts`)
	and one observed (`observed_readcounts`), with guides as columns and replicates as rows,
	computes the difference in mean log readcounts predicted from observed for each gene by 
	taking the mean of each sgRNA's difference of mean log readcounts. This is plotted vs
	the NB2 cost gradient on the gene's mean value. Genes with systematically higher predicted
	than observed readcounts should have negative cost gradients and vice versa. This is useful
	for Chronos debugging.
	'''
	estimated = pd.DataFrame(np.log10(predicted_readcounts.values+1), 
								 index=predicted_readcounts.index, 
								 columns=predicted_readcounts.columns
								)
	observed = pd.DataFrame(np.log10(observed_readcounts.values+1), 
								 index=observed_readcounts.index, 
								 columns=observed_readcounts.columns
								)
	estimated, observed = estimated.align(observed)
	diff = estimated.mean() - observed.mean()
	diff_gene = diff.groupby(guide_map.set_index("sgrna").gene).mean()
	density_scatter(diff_gene, ge_mean_grad,
					ax=ax, **density_scatter_args)
	plt.xlabel("Estimated - Observed Readcounts (Log10)")
	plt.ylabel("Mean Gene Effect Cost Gradient")


def check_integration_umap(gene_effect, sequence_map,
					variance_quantile=.5,
					 ax=None, metrics=None, legend=True, 
					  legend_args=dict(loc='upper left', bbox_to_anchor=(1, 1.05)),
						   scatter_args=dict(alpha=1, s=10)
					 ):
	'''
	Given the matrix of `gene_effect` and a `dict` of `sequence_map`s (see chronos.Chronos doc string
	for format), creates a UMAP embedding of cell lines in gene effect space, colored by the 
	presence of the cell lines in the various batches indicated by the keys of `sequence_map`. 
	To make the legend a manageable size, batch names are abbreviated to two letters. Returns the 
	max variance explained by batch membership as evaluated by finding the principle components of 
	`gene_effect`, correlating them with batch membership indicators, and multiplying that squared
	correlation with the variance explained by the component, summed over all PCs (one result
	per batch), returning the result for the batch that explains the most variance. This is useful
	to evaluate how well different batches are integrated.
	'''
	if not umap_present:
		raise ModuleNotFoundError("umap must be installed to use this plot")
	aliases = _make_aliases(list(sequence_map.keys()))
	palette = generate_powerset_palette(sequence_map.keys(), start=0, base_hsv_value=1)
	keysets = powerset(sequence_map.keys())
	keyset_lines = {}
	indicators = pd.DataFrame({
		key: pd.Series(True, 
					   index=sorted(set(sequence_map[key].cell_line_name) - set(['pDNA']))
					  )
		for key in sequence_map
	})
	indicators.fillna(False, inplace=True)
	gene_effect, indicators = gene_effect.align(indicators, join="inner", axis=0)
	sds = gene_effect.std()
	cutoff = sds.quantile(variance_quantile)
	gene_effect = gene_effect[sds.loc[lambda x: x>cutoff].index]
	for keyset in keysets:
		if not len(keyset):
			continue
		lines = set.intersection(*[set(sequence_map[key].cell_line_name) - set(['pDNA'])
								  for key in keyset]) & set(gene_effect.index)
		if len(keyset) < len(sequence_map):
			lines -= set.union(*[set(sequence_map[key].cell_line_name) - set(['pDNA'])
								  for key in sequence_map.keys()
								 if not key in keyset
								])
		keyset_lines[keyset] = sorted(lines)
	
	ump = UMAP(n_neighbors=5, min_dist=.02)
	umps = pd.DataFrame(ump.fit_transform(gene_effect.dropna(axis=1).values), index=gene_effect.index)
	for keyset, lines in keyset_lines.items():
		plt.scatter(umps.loc[lines, 0], umps.loc[lines, 1], label=''.join(aliases[list(keyset)]),
			color=palette[keyset],
				   linewidth=(len(keyset)-1)/2, edgecolor='black', **scatter_args)
	plt.xlabel("UMAP1")
	plt.ylabel("UMAP2")
	if legend:
		plt.legend(**legend_args)

	out = {}
	
	pca = PCA()
	pcs = pd.DataFrame(pca.fit_transform(gene_effect.dropna(axis=1).values), index=gene_effect.index)
	corrs_squared = fast_cor(pcs, indicators)**2
	out['library_pc_variance_explained_max'] = corrs_squared\
											.multiply(pca.explained_variance_ratio_, axis=0)\
											.sum()\
											.sort_values(ascending=False)\
											.max()
	if metrics is None:
		return out
	else:
		metrics.update(out)



def check_integration_mean_deviation(gene_effect, sequence_map,
					 ax=None, metrics=None, legend=True,
					  legend_args=dict(fontsize=7),
					plot_args=dict(lw=1)
					 ):
	'''
	Given the matrix of `gene_effect` and a `dict` of `sequence_map`s (see chronos.Chronos doc string
	for format), calculates the mean gene effect for each gene within each batch of the sequence map, 
	then the squared difference between that mean and the overall mean. This is plotted as a trend line
	per batch vs the overall gene mean. Returns the mean of the square root of this per-batch variance
	from the overall mean, and the genes with the largest variance in each batch.
	'''
	keyset_lines = {}
	indicators = pd.DataFrame({
		key: pd.Series(True, 
					   index=sorted(set(sequence_map[key].cell_line_name) - set(['pDNA']))
					  )
		for key in sequence_map
	})
	means1 = gene_effect.mean()
	cutoffs = means1.quantile([min(.1, 100/len(means1)), max(.9, 1-100/len(means1))])
	keep = means1.loc[lambda x: (x < cutoffs.iloc[1]) & (x > cutoffs.iloc[0])].index
	gene_effect = gene_effect[keep]
	indicators.fillna(False, inplace=True)
	gene_effect, indicators = gene_effect.align(indicators, join="inner", axis=0)
	library_means = pd.DataFrame({
		library: gene_effect[indicators[library]].mean()
		for library in indicators
	})
	means = gene_effect.mean()
	if ax is None:
		ax = plt.gca()
	else:
		plt.sca(ax)
	for library in indicators:
		y = (library_means[library]-means)**2
		trend = np.clip(lowess_trend(means, y), 0, np.inf)
		order = np.argsort(means)
		plt.plot(means.iloc[order], trend[order], label=library, **plot_args)
	plt.xlabel("Gene Mean Overall")
	plt.ylabel("Gene Mean Variance Trend")
	
	out = {}
	sd = gene_effect.std()
	normed_library_sd = np.sqrt(indicators.sum()) * np.abs(library_means.subtract(means, axis=0))
	out['normed_library_deviation'] = normed_library_sd.mean().mean()
	if legend:
		handles = append_to_legend_handles([
			"%s: %1.2f" % (key.replace("_", ' '), val)
			for key, val in out.items()
		], ax)
		plt.legend(handles=handles, **legend_args)
		
	out['library_outliers'] = {key: normed_library_sd[key].dropna().sort_values()[-5:]
							  for key in normed_library_sd}
	
	
	if ax is None:
		ax = plt.gca()
	else:
		plt.sca(ax)
	
	if metrics is None:
		return out
	else:
		metrics.update(out)


def guide_lfc_plot(lfc, palette):
	'''convenience method for kde plotting a subset of sgRNA's log fold change with fixed color for each sgRNA'''
	for j, key in enumerate(lfc.keys()):
		for guide in palette[key].index:
			sns.kdeplot(lfc[key][guide], label=key + guide[:4], bw_adjust=.5, color=palette[key][guide],
					   lw=.5)


def guide_palette(guide_map, gene):
	'''
	Returns a palette with a unique color for each sgRNA in `guide_map` targeting `gene`.
	'''
	start = np.pi * np.arange(len(guide_map))/len(guide_map)
	palette = {}
	for i, key in enumerate(guide_map):
		guides = guide_map[key].query("gene == %r" % gene).sgrna.unique()
		palette[key] = pd.Series(
			sns.cubehelix_palette(len(guides), start=start[i], rot=.25/len(guide_map), dark=.35, light=.7, hue=1),
			index=guides
		)
	return palette


def interrogate_gene(data, naive, naive_collapsed, gene, plot_width, plot_height):
	'''
	Creates a set of summary plots for a given gene effect profile.
	Parameters:
		`data` (`dict`): must contain (all of these files can be loaded from a `chronos.Chronos.save` directory)
			"gene_effect": `pandas.DataFrame` with genes as columns,
			"logfoldchange": `pandas.DataFrame` with sgRNAs as columns,
			"guide_efficacy": `pandas.Series` indexed by sgRNA with efficacy estimates,
			"t0_offset": `pandas.Series` indexed by sgRNA with offset estimates,
			"library effect": pandas.DataFrame` with genes as columns,
		`naive` (`dict`): contains a `pandas.DataFrame` matrix per batch with naive estimates of gene effect
			(typically median log fold change over guides per gene and replicates per cell line)
		`naive_collapsed`: a `pandas.DataFrame` matrix holding he consensus naive estimate over all libraries.
			Easily calculcated from `chronos.reports.collapse_dataframes`.
		`gene` (`str`): the gene of interest
		`plot_width`, `plot_height`: the total width of the figure and the height of individual panels, in inches.
	Returns:
		`matplotlib.Figure`
	'''
	palette = guide_palette(data['guide_map'], gene)
	fig, axes = plt.subplots(3, 2, figsize=(plot_width, plot_height*2.5))
	axes = [a for ax in axes for a in ax]

	plt.sca(axes[0])
	density_scatter(naive_collapsed[gene], data["gene_effect"][gene],
				   diagonal=True, label_outliers=5, outliers_from='diagonal')
	plt.xlabel("Naive Gene Effect")
	plt.ylabel("Gene Effect")
	
	plt.sca(axes[1])
	for j, key in enumerate(data['logfoldchange'].keys()):
		for guide in palette[key].index:
			sns.kdeplot(data['logfoldchange'][key][guide], label=key + '_' + guide[:4], bw_adjust=.5, 
						color=palette[key][guide],
					   lw=1)
	plt.legend(fontsize=6)
	plt.xlabel("Guide LFC")

	plt.sca(axes[2])
	labels = []
	for library in palette:
		x = data['guide_efficacy'].reindex(palette[library].index).fillna(-.1)
		y = data['t0_offset'][library].reindex(palette[library].index).fillna(-.1)
		plt.scatter(
			x, y,
			s=20, alpha=.75, linewidth=1, color=palette[library]
		)
		labels.extend([plt.text(s='%s_%s' % (library, ind[:4]), 
								x=x[ind],
								y=y[ind],
								fontsize=6, color=palette[library][ind]
							   ) for ind in palette[library].index])
	if adjustText_present:
		adjust_text(labels, arrowprops=dict(arrowstyle='-', color="black", lw=.5))
	plt.xlabel("Guide Efficacy")
	plt.ylabel("T0 Guide Offset")

	plt.sca(axes[3])
	x = pd.Series({library: naive[library][gene].mean()
		for library in naive
		if gene in naive[library]})
	y = data['library_effect'].loc[gene]
	colors = pd.Series({library: palette[library].iloc[0]
		for library in palette
		if len(palette[library])})
	x, y = x.dropna().align(y.dropna(), join="inner")
	x, colors = x.align(colors.dropna(), join="inner")
	x, y = x.align(y, join="inner")
	plt.scatter(x, y, s=20, alpha=.75, linewidth=1, c=colors)
	labels = [plt.text(
		s=ind, 
		x=x[ind],
		y=y[ind],
		fontsize=8, color=colors[ind]
	) for ind in x.index]
	if adjustText_present:
		adjust_text(labels, 
			arrowprops=dict(arrowstyle='-', color="black", lw=.5))
	plt.xlabel("Library Naive Gene Average")
	plt.ylabel("Library Effect")

	sorted_ge = data['gene_effect'][gene].sort_values().dropna()
	lowest_line = sorted_ge.index[0]
	highest_line = sorted_ge.index[-1]

	plt.sca(axes[4])
	single_line_interrogation(data, gene, lowest_line)
	plt.title('%s in %s (Lowest)' % (gene, lowest_line), fontsize=10)

	plt.sca(axes[5])
	single_line_interrogation(data, gene, highest_line)
	plt.title('%s in %s (Highest)' % (gene, highest_line), fontsize=10)

	return fig


def single_line_interrogation(data, gene, line, ax=None, 
		density_scatter_args={'trend_line': False, 'diagonal': True}
):
	'''
	A scatterplot of predicted vs observed log fold-change of sgRNAs for the selected gene
	in screened replicates of the selected line.
	Parameters:
		`data` (`dict`): see `interrogate_gene`
		`gene` (`str`): the gene of interest
		`line` (`str`): the cell line of interest.
	'''
	if not ax is None:
		plt.sca(ax)
	guides = {library: data['guide_map'][library].query("gene == %r" % gene).sgrna.unique()
				for library in data['guide_map']}
	sequences = {library: data['sequence_map'][library].query("cell_line_name == %r" % line).sequence_ID.unique()
				for library in data['sequence_map']
				}

	abbreviated_guide_mapper = {}
	abbreviated_replicate_mapper = {}
	stacked_lfc = []
	stacked_lfc_predicted = []
	aliases = _make_aliases(list(data['logfoldchange'].keys()))

	def consolidate_index(index):
		out = []
		for v in list(index):
			lib1, rep = v[0].split('Rep')
			lib2, guide = v[1].split('Guide')
			if lib1 != lib2:
				raise ValueError("Something went wrong in abbreviating index labels for log fold change")
			out.append( '%sRep%sGuide%s' % (lib1, rep, guide))
		return out

	for key, lfc in data['logfoldchange'].items():
		subset = lfc.loc[sequences[key], guides[key]]
		abbreviated_guide_mapper.update({guide: '%sGuide%i' %(aliases[key], i+1)
			for i, guide in enumerate(guides[key])})
		abbreviated_replicate_mapper.update({sequence: '%sRep%i' %(aliases[key], i+1)
			for i, sequence in enumerate(sequences[key])})
		subset_predicted = data['predicted_logfoldchange'][key].loc[sequences[key], guides[key]]
		subset.rename(index=abbreviated_replicate_mapper, columns=abbreviated_guide_mapper, inplace=True)
		subset_predicted.rename(index=abbreviated_replicate_mapper, columns=abbreviated_guide_mapper, inplace=True)
		stacked = subset.stack()
		stacked_predicted = subset_predicted.stack()
		stacked.index = consolidate_index(stacked.index)
		stacked_predicted.index = consolidate_index(stacked_predicted.index)
		stacked_lfc.append(stacked)
		stacked_lfc_predicted.append(stacked_predicted)
	x = pd.concat(stacked_lfc)
	y = pd.concat(stacked_lfc_predicted)
	x, y = x.align(y, join="inner")
	if not len(x):
		return
	density_scatter(x, y, **density_scatter_args)
	texts = [plt.text(s=ind, x=x[ind], y=y[ind], fontsize=7) for ind in x.index]
	if adjustText_present:
		adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=.5))
	plt.xlabel("Observed LFC")
	plt.ylabel("Predicted LFC")
	plt.title('%s in %s' % (gene, line))
	print("Guide and replicate key for %s, %s:\n%r\n%r\n%r" % (gene, line, aliases, pd.Series(abbreviated_guide_mapper), 
		pd.Series(abbreviated_replicate_mapper)))


def interrogate_gene_compare(paired_data, lfc, guide_map, gene, plot_width, plot_height):
	'''
	Creates a set of comparison plots for results from two different models for a specific gene.
	 This is mostly useful for internal Chronos development.
	Parameters:
		`paired_data` (`dict`): must contain two keys labeling `data` (`dict`) from two different models.
								See `interrogate_gene` for the format of `data`. 
		`lfc` (`dict`): one key per batch, with the value being a `pandas.DataFrame` of observed 
			log fold change, with sgRNAs as columns.
		`guide_map` (`pandas.DataFrame`): see `chronos.Chronos` for format.
		`gene` (`str`): the gene to examine.
		`plot_width`, `plot_height`: the total width of the figure and the height of individual panels, in inches.
	Returns:
		`matplotlib.Figure`
	'''
	keys = list(paired_data.keys())
	palette = guide_palette(guide_map, gene)
	fig, axes = plt.subplots(2, 2, figsize=(plot_width, plot_height))
	axes = [a for ax in axes for a in ax]
	
	plt.sca(axes[0])
	density_scatter(paired_data[keys[0]]["gene_effect"][gene], paired_data[keys[1]]["gene_effect"][gene],
				   diagonal=True, label_outliers=5, outliers_from='diagonal')
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	plt.title('%s Gene Effect' % gene)
	
	plt.sca(axes[2])
	for j, key in enumerate(lfc.keys()):
		for guide in palette[key].index:
			sns.kdeplot(lfc[key][guide], label=key + '_' + guide[:4], bw_adjust=.5, 
						color=palette[key][guide],
					   lw=1)
	plt.legend(fontsize=6)
	plt.xlabel("Guide LFC")
	
	plt.sca(axes[3])
	plt.title("Guide Efficacy")
	labels = []
	for library in lfc:
		x = paired_data[keys[0]]['guide_efficacy'].reindex(palette[library].index).fillna(-.1)
		y = paired_data[keys[1]]['guide_efficacy'].reindex(palette[library].index).fillna(-.1)
		plt.scatter(
			x, y,
			s=20, alpha=.75, linewidth=1, color=palette[library]
		)
		labels.extend([plt.text(s='%s_%s' % (library, ind[:4]), 
								x=x[ind],
								y=y[ind],
								fontsize=6, color=palette[library][ind]
							   ) for ind in palette[library].index])
	if adjustText_present:
		adjust_text(labels, arrowprops=dict(arrowstyle='-', color="black", lw=.5))
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	
	plt.sca(axes[1])
	corrs = {}
	for key in keys:
		corrs[key] = {}
		for library in lfc:
			naive = lfc[library][palette[library].index]\
					.groupby(paired_data[key]['sequence_map'][library].set_index("sequence_ID").cell_line_name)\
					.median()
			series = fast_cor(
				paired_data[key]['gene_effect'][[gene]], 
				naive
			).loc[gene]
			corrs[key][library] = series
	labels = []
	for library in lfc:
		plt.scatter(corrs[keys[0]][library], corrs[keys[1]][library],
				   color=palette[library], s=15, alpha=.75)
		labels.extend([plt.text(s='%s_%s' % (library, ind[:4]), 
								x=corrs[keys[0]][library][ind],
								y=corrs[keys[1]][library][ind],
								fontsize=6, color=palette[library][ind]
							   ) for ind in corrs[keys[0]][library].index])
	if adjustText_present:
		adjust_text(labels, arrowprops=dict(arrowstyle='-', color="black", lw=.5))
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	plt.title("Gene Effect - Guide LFC Corr")
	
	return fig