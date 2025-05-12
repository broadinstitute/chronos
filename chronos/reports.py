try:
	import reportlab
except ModuleNotFoundError:
	raise ModuleNotFoundError("reportlab must be installed to use the reports module. Try `pip install reportlab`")
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from .model import read_hdf5, calculate_fold_change, powerset, normalize_readcounts
from .evaluations import *

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

from .plotting import density_scatter, dict_plot
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection


def load_chronos_data_for_qc(directory, gene_effect_file="gene_effect.hdf5"):
	'''
	Loads the results of a Chronos run saved to the `directory` using the `Chronos.save` method in a `dict`
	suitable for passing to qc report functions.
	Parameters:
		`directory` (`str`): location of the saved run
		`gene_effect_file` (`str`): optionally specify a different file in the directory where gene effect is
			saved. This can be used to load a copy-mumber corrected version of the data. Must be in Chronos'
			h5 format.
	Returns:
		`dict` containing the results of the run with the keys expected by the qc report functions in this module.
	'''
	libraries = [
		f.split('_')[0]
		for f in os.listdir(directory)
		if f.endswith("sequence_map.csv")
	]
	data = {
		'gene_effect': read_hdf5(os.path.join(directory, gene_effect_file)),
		'library_effect': pd.read_csv(os.path.join(directory, "library_effect.csv"), index_col=0),
		't0_offset': pd.read_csv(os.path.join(directory, "t0_offset.csv"), index_col=0),
		'guide_efficacy': pd.read_csv(os.path.join(directory, "guide_efficacy.csv"), index_col=0)["efficacy"],
		'cell_line_efficacy': pd.read_csv(os.path.join(directory, "cell_line_efficacy.csv"), index_col=0),
		'growth_rate': pd.read_csv(os.path.join(directory, "cell_line_growth_rate.csv"), index_col=0),
		'readcounts': {
			library: read_hdf5(os.path.join(directory, "%s_readcounts.hdf5" % library))
			for library in libraries
		},
		'sequence_map': {
			library: pd.read_csv(os.path.join(directory, "%s_sequence_map.csv" % library))
			for library in libraries
		},
		'guide_map': {
			library: pd.read_csv(os.path.join(directory, "%s_guide_gene_map.csv" % library))
			for library in libraries
		},
		'excess_variance': {
			library: pd.read_csv(os.path.join(directory, "screen_excess_variance.csv"), index_col=0)[library]
			for library in libraries
		},
		'predicted_readcounts': {
			library: read_hdf5(os.path.join(directory, "%s_predicted_readcounts.hdf5" % library))
			for library in libraries
		},
		'predicted_logfoldchange': {
			library: read_hdf5(os.path.join(directory, "%s_predicted_lfc.hdf5" % library))
			for library in libraries
		},

	}

	data["logfoldchange"] = {}
	for library in libraries:
		fc = calculate_fold_change(
				data["readcounts"][library],
				data["sequence_map"][library],
				rpm_normalize=False
		)
		data['logfoldchange'][library] = pd.DataFrame(
			np.log2(fc.values),
			index=fc.index, columns=fc.columns
		)
	return data


def get_naive(data):
	'''
	Computes naive gene effect per library libraries by finding the median 
	of guides/gene and replicates/line within each library
	Parameters:
		`data` (`dict`): must have keys "logfoldchange", "guide_map", and "sequence_map"
	returns:
		`dict`[`pandas.DataFrame`] holding naive gene effect estimates.
	'''
	naive = {}
	for library in data["logfoldchange"]:
		naive[library] = data['logfoldchange'][library]\
			.T.groupby(data['guide_map'][library].set_index("sgrna").gene)\
			.median().T\
			.groupby(data['sequence_map'][library].set_index("sequence_ID").cell_line_name)\
			.median()
	return naive


def mean_collapse_dataframes(dfs):
	'''
	Given an iterable of pandas DataFrames, returns a single dataframe
	where each value is given by the mean value for the same index/column
	across the input DataFrames, ignoring NaNs.
	'''
	numerator = None
	denominator = None
	for df in dfs:
		if numerator is None:
			numerator = df.fillna(0)
			denominator = df.notnull().astype(int)
		else:
			numerator, df = numerator.align(df, join='outer')
			numerator.fillna(0, inplace=True)
			denominator, numerator = denominator.align(numerator, join="right")
			denominator.fillna(0, inplace=True)
			numerator += df.fillna(0).values
			denominator += df.notnull().values
	numerator = numerator.mask(denominator==0)
	denominator.replace(0, np.nan, inplace=True)
	return numerator/denominator
	

def qc_compare_plot(plot_func, data, data_key, metrics, plot_width, plot_height, **kwargs):
	'''
	A convenience method for comparing results from two different runs side by side
	Parameters:
		`plot_func` (`function`): a plotting function that accepts an object of the type `data[data_key]`
			and a `metrics` kew word argument and plots to the current matplotlib axis
		`data` (`dict`): dict containing data to plot
		`data_key` (`str`): the entry in the `data` that will be plotted
		`metrics` (`dict`): passed to `plot_func`
		`plot_width`, `plot_height`: desired (total) plot size in inches
		Additions kwargs passed to `plot_func`
	Returns:
		`matplotlib.Figure`
	'''
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	for i, key, in enumerate(data.keys()):
		plt.sca(axes[i])
		plt.title(key)
		plot_func(data[key][data_key], metrics=metrics[key], **kwargs)
	plt.tight_layout()
	return fig


def qc_initial_data(title, readcounts, sequence_map, guide_map, negative_control_sgrnas=None, positive_control_sgrnas=None,
		   report_name=None, directory='./', plot_width=7.5, plot_height=3.25,
		  doc_args=dict(
			pagesize=letter, rightMargin=.5*inch, leftMargin=.5*inch,
			topMargin=.5*inch,bottomMargin=.5*inch
		  ),
		  specific_plot_dimensions={}
):
	'''
	QC dthe data that would be passed to Chronos. This can be helpful to develop a sense of data quality but also to exclude 
	bad results.
	Parameters:
		`title` (`str`): the report title, printed on first page
		`readcounts` (`pd.DataFrame`): read numbers for each pDNA and late timepoint as rows with sgRNAs as columns.
		 	Do not need to be normalized.
		`sequence_map` (`pd.DataFrame`): map of sequences for both pDNA and late replicates to cell lines, timepoints, and pDNA batches.
			See `chronos.Chronos` for format.
		`guide_map` (`pd.DataFrame`): map of sgRNAs to genes.  Must include the columns 'sgrna' and 'gene'.
		`negative_control_sgrnas`, `positive_control_sgrnas` (ordered indexable of `str`): optional guides where no effect or
			a strong depleting effect is expected, respectively. If not provided a number of the more useful QC metrics can't
			be calculated.
		`report_name` (`str`): an optional file name for the report. If none is provided, `title` + '.pdf' will be used.
		`directory` (`str`): where the report and figure panels will be generated.
		`plot_width`, `plot_height` (`float`): size of plots that will be put in the report in inches.
		`doc_args` (`dict`): additional arguments will be passed to `SimpleDocTemplate`.
		`specific_plot_dimensions` (`dict` of 2-tuple`): if a plot's name is present, will use the the value
			 to specify dimensions for that plot instead of deriving them from `plot_width` and `plot_height`
	Returns:
		`dict` containing the calculated QC metrics, which will also be in the report.
	'''
	if report_name is None:
		report_name = title + ".pdf"
	doc = SimpleDocTemplate(os.path.join(directory, report_name), **doc_args)
	styles=getSampleStyleSheet()
	story = []
	metrics = {}
	
	def add_image(filename):
		fig = plt.gcf()
		label = '.'.join(filename.split('.')[:-1])
		if label in specific_plot_dimensions:
			fig.set_size_inches(specific_plot_dimensions[label])
		width, height = fig.get_size_inches()
		plt.tight_layout()
		fig.savefig(os.path.join(directory, filename))
		plt.close(fig)
		im = Image(os.path.join(directory, filename), width*inch, height*inch)
		story.append(im)
		story.append(Spacer(.125, 12))
			
	normalized = normalize_readcounts(readcounts, negative_control_sgrnas, sequence_map)
	lfc = np.log2(calculate_fold_change(normalized, sequence_map,rpm_normalize=False))
	nlines = len(set(sequence_map.cell_line_name) - set(['pDNA']))
	
	print("calculating replicate correlation")
	mean_corrs = []    
	for line in sequence_map.cell_line_name.unique():
		if line == 'pDNA':
			continue
		reps = sequence_map.query("cell_line_name == %r" % line).sequence_ID
		corrs = fast_cor(lfc.loc[reps].T)
		np.fill_diagonal(corrs.values, np.nan)
		mean_corrs.append(corrs.mean())
	metrics['MeanReplicateCorr'] = pd.concat(mean_corrs)
	metrics["ReplicateCorrWithMean"] = lfc.corrwith(lfc.mean(), axis=1)
	worst = metrics['MeanReplicateCorr']\
				.groupby(sequence_map.set_index("sequence_ID").cell_line_name)\
				.min()\
				.sort_values().dropna().index[:10]
	
	def get_nnmd(x):
		return nnmd(x[positive_control_sgrnas], x[negative_control_sgrnas])
	def get_roc_auc_score(x):
		return auroc(x[positive_control_sgrnas], x[negative_control_sgrnas])

	if not negative_control_sgrnas is None and not positive_control_sgrnas is None:
		print("generating control separation metrics")
		negative_control_sgrnas = sorted(set(negative_control_sgrnas) & set(readcounts.columns))
		if not len(negative_control_sgrnas):
			raise ValueError(
				"none of the negative control sgRNAs found in readcounts columns:\n%r" 
				% negative_control_sgrnas
			)
		positive_control_sgrnas = sorted(set(positive_control_sgrnas) & set(readcounts.columns))
		if not len(positive_control_sgrnas):
			raise ValueError(
				"none of the negative control sgRNAs found in readcounts columns:\n%r" 
				% positive_control_sgrnas
			)
		metrics['NNMD'] = lfc.apply(get_nnmd, axis=1)
		metrics['AUROC'] = lfc.apply(get_roc_auc_score, axis=1)
		metrics["PosConMedian"] = lfc[positive_control_sgrnas].median(axis=1)
		metrics["NegConMedian"] = lfc[negative_control_sgrnas].median(axis=1)
		metrics["NegConSD"] = lfc[negative_control_sgrnas].std(axis=1)
		worst_sep = metrics['AUROC']\
				.groupby(sequence_map.set_index("sequence_ID").cell_line_name)\
				.min()\
				.sort_values().dropna().index[:10]
		worst = sorted(set(worst) & set(worst_sep))
	
	else:
		print("One or both control groups not supplied, skipping control separation metrics")
	story.append(Paragraph(title, style=styles["Heading1"]))
	
	print("Plotting log fold-change distribution")
	story.append(Paragraph("sgRNA Log Fold-Change Distribution", style=styles["Heading2"]))
	story.append(Paragraph(
"For a traditional genome-wide loss of viability experiment we expect the bulk of log fold change \
scores near 0, with a long left tail of true viability depletion."
	))
	
	sns.kdeplot(lfc.stack(), label="All sgRNAs", fill=True, color="gray", bw_adjust=.25)
	if not negative_control_sgrnas is None:
	   sns.kdeplot(lfc[negative_control_sgrnas].stack(), label="Negative Controls sgRNAs", 
				   color=[.3, .1, .9], bw_adjust=.25)
	if not positive_control_sgrnas is None:
		sns.kdeplot(lfc[positive_control_sgrnas].stack(), label="Positive Controls sgRNAs", 
				   color=[.9, .2, 0], bw_adjust=.25)
	plt.legend()
	plt.xlabel("Log Fold-Change of late timepoints from pDNA")
	plt.gcf().set_size_inches((plot_width, plot_height))
	add_image("lfc_distribution.png")
	
	if 'NNMD' in metrics:
		print("plotting control separation metrics")
		story.append(Paragraph("Control QC Metrics", style=styles["Heading2"]))
		story.append(Paragraph(
"Depletion of positive controls is a positive signal for screen quality, while \
high standard deviation in negative controls is a negative signal for screen quality. \
However, these measures tend to be negatively correlated in CRISPR screens: screens that show \
the greatest dropout of essential genes also have the greatest noise in nonessential genes."
		))
		
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
		
		plt.sca(axes[0])
		density_scatter(metrics["PosConMedian"] - metrics["NegConMedian"],
								 metrics["NegConSD"], 
								 label_outliers=4,
								alpha=.5)
		plt.xlabel("Pos. Con. median LFC")
		plt.ylabel("Neg. Con. SD")
		
		story.append(Paragraph(
"The null-normalized median difference (NNMD) is"
		))
		story.append(Paragraph(
			"\t\t((median(positive controls) - median(negative controls)) / mad(negative controls)"
		))
		story.append(Paragraph(
"In Project Achilles, we look for NNMD scores below -1.25 to consider a replicate passing \
but this threshold depends strongly on the controls you have chosen. \
We also provide the area under the ROC curve for separating the positive and negative control \
log fold changes. These measures should have a strong negative correlation."
		))
		plt.sca(axes[1])
		density_scatter(metrics["NNMD"], metrics["AUROC"], label_outliers=4, outliers_from="xy_zscore",
								alpha=.5)
		xlim, ylim = plt.gca().get_xlim(), plt.gca().get_ylim()
		plt.xlabel("NNMD")
		plt.ylabel("AUROC")
		
		
		add_image("control_sep.png")

	if metrics["MeanReplicateCorr"].any():
		story.append(Paragraph("Replicate Correlation", style=styles["Heading2"]))
		story.append(Paragraph(
"Below is the Pearson correlation of replicate Log Fold-Change with the mean LFC over all replicates (x axis) vs \
the mean correlation with other replicates of the same cell line (y axis). Generally these are closely related \
and correlate with other measures of screen quality."))
		density_scatter(metrics["ReplicateCorrWithMean"], metrics["MeanReplicateCorr"],
					   label_outliers=5)
		plt.xlabel("Replicate R with Mean LFC")
		plt.ylabel("Mean Replicate R with same line")
		add_image("replicate_correlations.png")
		
	story.append(PageBreak())
	story.append(Paragraph("Details for worst performing cell lines", style=styles["Heading2"]))
	story.append(Paragraph(
"For a dozen or so of the lines with the worst quality metrics, more details are given below. \
It can be useful to look at the replicate-replicate plots carefully for effects such as"
	))
	story.append(Paragraph("\t- dropouts that aren't shared between replicates"))
	story.append(Paragraph(
		"\t- extreme outgrowths (whether shared or not). \
These are concerning unless there is a sound biological reason \
such as tumor suppressor KO or your experiment is a rescue experiment."
	 ))
	story.append(Paragraph(""))
	story.append(Paragraph(
"We also show reads in the late timepoints compared to the pDNA. If control groups are provided, these are broken \
out separately. We expect negative control sgRNAs to be closely aligned to pDNA abundance, while positive control \
sgRNAs should tend to fall below the diagonal. Note that each axis is the log(normalized counts + 1)."))
	for line in worst:
		story.append(PageBreak())
		story.append(Paragraph(line, style=styles["Heading3"]))
		all_replicate_plot(normalized, sequence_map, line, plot_width)
		add_image("%s_rep_plot.png" % line)
		paired_pDNA_plots(normalized, sequence_map, line, negative_control_sgrnas, positive_control_sgrnas,
						 plot_width, plot_height)
		add_image("%s_pdna_plot.png" % line)
		
	doc.build(story)
	
	return metrics


def dataset_qc_report(title, data,
	positive_control_genes, negative_control_genes, 
	mutation_matrix=None, addiction_expressions=None, copy_number=None,
	report_name=None, directory='.', gene_effect_file="gene_effect.hdf5",
						  plot_width=7.5, plot_height=3.25,
						  doc_args=dict(
							pagesize=letter, rightMargin=.5*inch, leftMargin=.5*inch,
							topMargin=.5*inch,bottomMargin=.5*inch
						  ),
						  specific_plot_dimensions={}
):
	'''
	QC the results of the Chronos run.
	Parameters:
		`title` (`str`): the report title, printed on first page
		`data` (`str` or `dict`): A path to a saved Chronos directory, or the results of `load_chronos_data_for_qc`. 
			If you manually assemble `data` as a `dict`, please consult that function for the correct format.
		`positive_control_genes`, `negative_control_genes` (`list`, `pandas.Index`, or `numpy.array` of `str`):
			Genes whose KO is expected to cause loss of viability or no loss of viability, respectively.
		`mutation_matrix` (`pandas.DataFrame`): optional boolean matrix of cell line by gene.
			Each value indicates that the gene has a gain of function mutation in that cell line.
			Genes should be selected such that a gain of function mutation is expected to make the cell line
			dependent on that gene. Tbhis is used to evaluate the separation of gene effects for that gene
			between mutated and wildtype cell lines.
		`addiction_expressions` (`pandas.DataFrame`): optional `float` matrix of cell lines by genes containing
			expressions. The genes should be chosen such that cell lines highly expressing the gene are expected
			to be dependent on it, while other cell lines are not.
		`copy_number` (`pandas.DataFrame`): optional cell line by gene `float` matrix of logged copy number counts. Used to QC the copy
			number effect. 
		`report_name` (`str`): an optional file name for the report. If none is provided, `title` + '.pdf' will be used.
		`directory` (`str`): where the report and figure panels will be generated.
		`gene_effect_file` (`str`): If `data` is a path to a directory, this arg is passed to `load_chronos_data_for_qc`.
		`plot_width`, `plot_height` (`float`): size of plots that will be put in the report in inches.
		`doc_args` (`dict`): additional arguments will be passed to `SimpleDocTemplate`.
		`specific_plot_dimensions` (`dict` of 2-tuple`): if a plot's name is present, will use the the value
			 to specify dimensions for that plot instead of deriving them from `plot_width` and `plot_height`
	Returns:
		`dict` containing the calculated QC metrics, which will also be in the report.
	'''
	if isinstance(data, str):
		try:
			print("Loading data from %s" % data)
			data = load_chronos_data_for_qc(data, gene_effect_file)
		except IOError:
			raise ValueError("If `data` is a string, it must be the path to a directory containing Chronos saved data. \
gene_effect_file must be the name of an hdf5 file in that directory. \
You passed '%s', %r" % (data, gene_effect_file))
	if not isinstance(data, dict):
		raise ValueError("`data` must be a `dict` of data or a string pointing to Chronos saved directory")
	required_data_keys = ["gene_effect", "sequence_map", "guide_map", "guide_efficacy",
						  "predicted_readcounts", "readcounts",
						 "logfoldchange", 'predicted_logfoldchange', 
						 "excess_variance", "growth_rate", "cell_line_efficacy",
						 "t0_offset", "library_effect"
						 ] 
	for key in required_data_keys:
		if not key in data:
			raise ValueError("`data` missing required entry %s" % (key))
	library_data = {
		library: {
			key: data[key][library]
			for key in ['readcounts', 'predicted_readcounts', 
						'logfoldchange', 'predicted_logfoldchange',
						"excess_variance"
					   ]
		}
		for library in data['readcounts']
	}
	orig_working_dir = os.getcwd()
	if report_name is None:
		report_name = title + ".pdf"
	doc = SimpleDocTemplate(os.path.join(directory, report_name), **doc_args)
	styles=getSampleStyleSheet()
	story = []
	metrics = {}


	def add_image(filename):
		fig = plt.gcf()
		label = '.'.join(filename.split('.')[:-1])
		if label in specific_plot_dimensions:
			fig.set_size_inches(specific_plot_dimensions[label])
		width, height = fig.get_size_inches()
		plt.tight_layout()
		fig.savefig(os.path.join(directory, filename))
		plt.close(fig)
		im = Image(os.path.join(directory, filename), width*inch, height*inch)
		story.append(im)
		story.append(Spacer(.125, 12))

	
	story.append(Paragraph(title, style=styles["Heading1"]))
	
	story.append(Paragraph("Control Separation", style=styles["Heading2"]))
	print("plotting global control separation")
	story.append(Paragraph("Global Control Separation", style=styles["Heading3"]))
	story.append(Paragraph(
"Separation of positive/negative control genes both overall and by screen. \
More negative NNMD is better."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	control_histogram(data["gene_effect"], positive_control_genes,
					negative_control_genes, metrics=metrics)
	plt.sca(axes[1])
	screen_nnmd_auroc_scatter(data["gene_effect"], positive_control_genes,
					negative_control_genes, metrics=metrics)
	add_image("global_controls.png")
	
	if (not mutation_matrix is None) or (not addiction_expressions is None):
		print("plotting selective dependency separation")
		story.append(Paragraph("Selective Control Separation", style=styles["Heading3"]))
		story.append(Paragraph(
"Separation of known selective dependencies between indications. \
On the left, known oncogene gene effects are compared between models where \
a known oncogenic GoF mutation occurred in that gene vs the rest, if `mutation_matrix` is supplied. \
On the right, we test expression addictions using a one-tailed test on pearson correlations, \
if `addiction_expressions` is supplied. \
The FDRs should be considered optimistic."
		))
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
		plt.sca(axes[0])
	if not mutation_matrix is None:
		selective_mutated_vs_not_scatter(data["gene_effect"], mutation_matrix, metrics=metrics)
	plt.sca(axes[1])
	if not addiction_expressions is None:
		expression_addiction_volcano(data["gene_effect"], addiction_expressions, metrics=metrics)
	if (not mutation_matrix is None) or (not addiction_expressions is None):
		add_image("selective_dependencies.png")
	story.append(PageBreak())
	

	story.append(Paragraph("General Parameter Info", style=styles["Heading2"]))

	story.append(Paragraph("Statistical Properties of Gene Effects", style=styles["Heading3"]))
	print("plotting gene effect mean relationships")
	story.append(Paragraph(
"Higher overall gene SD is better (if control separation in each cell line is maintained). There is usually a trend \
towards more variance in more negative genes. There should NOT be a trend in the second plot."
))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	mean_vs_sd_scatter(data["gene_effect"], metrics=metrics)
	plt.sca(axes[1])
	mean_vs_cell_eff_correlation(data['gene_effect'], data['cell_line_efficacy'].mean(axis=1))
	add_image("gene_effect_properties.png")

	if not copy_number is None:
		print("plotting copy number effect")
		story.append(Paragraph("Copy Number Effect", style=styles["Heading3"])) 
		story.append(Paragraph(
		"Relationship of genomic copy number to estimated gene effect both overall (left) and per gene binned \
		by gene mean (right). Ideally there is no systematic relationship."
		))
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
		plt.sca(axes[0])
		copy_number_trend(data['gene_effect'], copy_number, downsample=.01, downsample_lower_quantile_bound=.01,
						downsample_upper_quantile_bound=.99, metrics=metrics)
		plt.sca(axes[1])
		copy_number_gene_corrs(data['gene_effect'], copy_number, metrics=metrics)
		add_image("copy_number_effect.png")
	
	print("plotting screen efficacy and growth rate")
	story.append(Paragraph("Screen Efficacy, Growth Rate, and Guide Efficacy", style=styles["Heading3"]))
	story.append(Paragraph(
"These parameters together translate a gene effect into the expected impact on cell proliferation. \
Often there will be a trend towards lower growth estimates with lower cell efficacy estimates. \
Guide efficacies have a single global value, but here have been grouped by presence in a library. \
They should have a high peak near 1."))
	growth_rate = []
	cell_line_efficacy = []
	for library in library_data:
		gr, cle = data["growth_rate"][library].dropna().align(data['cell_line_efficacy'][library].dropna(), join="inner")
		growth_rate.append(gr)
		cell_line_efficacy.append(cle)
	growth_rate, cell_line_efficacy = pd.concat(growth_rate), pd.concat(cell_line_efficacy)
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	density_scatter(growth_rate, cell_line_efficacy, trend_line=False, label_outliers=4, outliers_from="xy_zscore")
	plt.xlabel("Relative Growth Rate")
	plt.ylabel("Cell Line Efficacy")
	metrics["growth_rate_sd"] = growth_rate.std()
	metrics["cell_efficacy_mean"] = cell_line_efficacy.mean()
	plt.sca(axes[1])
	for library, guide_map in data['guide_map'].items():
		guides = guide_map.sgrna.unique()
		efficacies = data['guide_efficacy'].reindex(guides).dropna()
		sns.kdeplot(efficacies, bw_adjust=.5, lw=1, label=library)
		metrics["guide_eff_%s_mean" % library] = efficacies.mean()
	plt.legend()
	plt.xlabel("Guide Efficacy")
	add_image("parameter_distributions.png")
	story.append(PageBreak())

	if len(data['guide_map']) > 1:
		print("plotting library integration")
		story.append(Paragraph("Library Integration", style=styles["Heading2"]))
		story.append(Paragraph(
			"The UMAP embedding of cell line gene effects colored by library presence (left) and how \
	far a gene's average within a library deviates from the overall average, by library (right). \
	The UMAP embedding uses only the 50% most variable genes. \
	On the right, a lowess trend is fitted per library to the squared difference of the gene's mean within \
	models screened with the library and its mean overall."
		))
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
		plt.sca(axes[0])
		check_integration_umap(data['gene_effect'], data['sequence_map'], metrics=metrics)
		plt.sca(axes[1])
		check_integration_mean_deviation(data['gene_effect'], data['sequence_map'], metrics=metrics)
		story.append(Paragraph("Prediction Accuracy", style=styles["Heading2"])) 
		add_image("library_integration.png")
		story.append(PageBreak())

	print("plotting readcount predictions")
	story.append(Paragraph("Predictions", style=styles["Heading2"]))
	story.append(Paragraph("Readcount Predictions", style=styles["Heading3"]))
	story.append(Paragraph(
"Chronos' readcount predictions should generally line up well with observation, but it will predict \
greater than observed readcounts for cases with very few counts."
	))

	def plot_func(x):   
		predicted_vs_observed_readcounts(
			x["predicted_readcounts"], x['readcounts'],
						metrics=metrics)
	fig, axes = dict_plot(library_data, plot_func, plot_width)
	add_image("readcount_predictions.png")
	
	print("plotting LFC predictions")
	story.append(Paragraph("Log Fold-Change Predictions", style=styles["Heading3"]))
	story.append(Spacer(.125, 12))
	story.append(Paragraph(
"Screens with greater excess variance (overdispersion) should have worse correlation between \
observed LFC and Chronos' predictions."
	))
	def plot_func(x):
		lfc_corr_vs_excess_variance(
			x["predicted_logfoldchange"], x['logfoldchange'], x['excess_variance'],
						metrics=metrics)
	fig, axes = dict_plot(library_data, plot_func, plot_width)
	add_image("lfc_corr_vs_excess_variance.png")
	story.append(PageBreak())
	

	print("plotting difference from naive gene score")
	naive = get_naive(data)
	naive_collapsed = mean_collapse_dataframes(naive.values())
	story.append(Paragraph("Gene Score Difference from Naive", style=styles["Heading2"]))
	story.append(Paragraph(
		"Comparing the gene effect scores to a naive score estimated as log fold change median per guide/replicate \
within libraries, then the mean across libraries. The first plots show the correlation of individual genes, both vs mean effect \
and vs the difference of means between \
the supplied and naive gene effects. Below is the direct comparison of gene means and a comparison of the most extreme \
values for each gene's score."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	gene_corr_vs_mean(naive_collapsed, data['gene_effect'],
					metrics=metrics)
	plt.sca(axes[1])
	gene_corr_vs_mean_diff(naive_collapsed, data['gene_effect'],
					metrics=metrics)
	plt.xlabel("Naive Mean - Gene Effect Mean")
	add_image("gene_corrs.png")

	fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_width - 2))
	plt.sca(ax)
	density_scatter(naive_collapsed.mean(), data['gene_effect'].mean(), diagonal=True, 
					label_outliers=10, alpha=.5, s=10)
	plt.title("Mean Gene Effect")
	plt.xlabel("Naive")
	plt.ylabel("Gene Effect")
	add_image("gene_means.png")
	fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_width - 2))
	plt.sca(ax)
	gene_outlier_plot(naive_collapsed, data['gene_effect'], metrics=metrics)
	plt.title("Most Extreme Z-Scores by Gene")
	plt.xlabel("Gene Effect Extreme ZScore")
	plt.ylabel("Naive Extreme ZScore")
	add_image("gene_zscore_extremes.png")
	story.append(PageBreak())
	
	print("summarizing")
	ge_mean = data['gene_effect'].mean()
	cell_line_mean = data['gene_effect'].mean(axis=1).std()/ge_mean.std()
	naive_means = {key: v.mean() for key, v in naive.items()}

	naive_corr_text = '\n'.join([
		'\t%s: %1.3f' % (key, v.corr(ge_mean))
		for key, v in naive_means.items()
	])
	story.insert(1, Paragraph(
'''
Summary: the standard deviation (SD) of gene means in gene effect is %1.3f.\n
The mean of gene SDs is %1.3f the SD of gene means.\n
The SD of cell line means is %1.3f the SD of gene means\n. 
The correlation of each library's mean LFC per gene with Chronos' mean gene effect is:\n
%s
''' % (ge_mean.std(), metrics['mean_SD:SD_means'], cell_line_mean, naive_corr_text)
	))

	print("plotting genes with low agreement with naive gene effect")
	story.append(Paragraph("Exploring Low Agreement Genes", style=styles['Heading2']))
	story.append(Spacer(.125, 12))
	story.append(Paragraph("In the remaining plots, the genes with lowest agreement are explored further. \
NA results for guide efficacy are replaced with -.1"))
	story.append(Spacer(.125, 12))

	outliers = set(metrics['worst_agreement']) \
				| set([s.split('_')[0] for s in metrics['low_outliers']]) \
				| set([s.split('_')[0] for s in metrics['high_outliers']])
	for gene in outliers:
		print("\t%s" % gene)
		header = Paragraph(gene, style=styles["Heading3"])
		story.append(header)
		fig = interrogate_gene(data, naive, naive_collapsed, gene, plot_width, plot_height)
		add_image(gene + '.png')
		story.append(PageBreak())
			

	print("building report")
	doc.build(story)
	return metrics




def comparative_qc_report(title, data, 
						  positive_control_genes, negative_control_genes, 
						  mutation_matrix, addiction_expressions,
						  report_name=None, directory='.', 
						  plot_width=7.5, plot_height=3.25,
						  doc_args=dict(
							pagesize=letter, rightMargin=.5*inch, leftMargin=.5*inch,
							topMargin=.5*inch,bottomMargin=.5*inch
						  ),
						  specific_plot_dimensions={}
):
	'''
	Compare the output of two Chronos runs, or Chronos with another algorithm (if that algorithm also 
		estimates gene effect and guide efficacy). 
	Parameters:
		`title` (`str`): the report title, printed on first page
		`data` (`dict`): A `dict` with EXACTLY two entries. the keys of the entries will be used as labels
			in the plots in the report. Each value is also a `dict` which must contain the keys 'gene_effect',
			'sequence_map', 'guide_map', 'guide_efficacy', and 'logfoldchange'. Gene effect and guide efficacy
			are model outputs, while logfoldchange can be calculated directly from the data.
		`positive_control_genes`, `negative_control_genes` (`list`, `pandas.Index`, or `numpy.array` of `str`):
			Genes whose KO is expected to cause loss of viability or no loss of viability, respectively.
		`mutation_matrix` (`pandas.DataFrame`): optional boolean matrix of cell line by gene.
			Each value indicates that the gene has a gain of function mutation in that cell line.
			Genes should be selected such that a gain of function mutation is expected to make the cell line
			dependent on that gene. Tbhis is used to evaluate the separation of gene effects for that gene
			between mutated and wildtype cell lines.
		`addiction_expressions` (`pandas.DataFrame`): optional `float` matrix of cell lines by genes containing
			expressions. The genes should be chosen such that cell lines highly expressing the gene are expected
			to be dependent on it, while other cell lines are not.
		`copy_number` (`pandas.DataFrame`): optional cell line by gene `float` matrix of logged copy number counts. Used to QC the copy
			number effect. 
		`report_name` (`str`): an optional file name for the report. If none is provided, `title` + '.pdf' will be used.
		`directory` (`str`): where the report and figure panels will be generated.
		`gene_effect_file` (`str`): If `data` is a path to a directory, this arg is passed to `load_chronos_data_for_qc`.
		`plot_width`, `plot_height` (`float`): size of plots that will be put in the report in inches.
		`doc_args` (`dict`): additional arguments will be passed to `SimpleDocTemplate`.
		`specific_plot_dimensions` (`dict` of 2-tuple`): if a plot's name is present, will use the the value
			 to specify dimensions for that plot instead of deriving them from `plot_width` and `plot_height`
	Returns:
		`dict` containing the calculated QC metrics, which will also be in the report.
	'''
	required_data_keys = ["gene_effect", "sequence_map", "guide_map", "guide_efficacy",
						 "logfoldchange"]
	if len(data) != 2:
		raise ValueError("`data` must be a dict with two keys")
	for key, val in data.items():
		for key2 in required_data_keys:
			if not key2 in data[key]:
				raise ValueError("`data[%s] missing required entry %s" % (key, key2))

	if report_name is None:
		report_name = title + ".pdf"
	
	doc = SimpleDocTemplate(os.path.join(directory, report_name), **doc_args)
	styles=getSampleStyleSheet()
	keys = list(data.keys())
	story = []
	metrics = {keys[0]: {}, keys[1]: {}, "joint": {}}

	def add_image(filename):
		fig = plt.gcf()
		label = '.'.join(filename.split('.')[:-1])
		if label in specific_plot_dimensions:
			fig.set_size_inches(specific_plot_dimensions[label])
		width, height = fig.get_size_inches()
		plt.tight_layout()
		fig.savefig(os.path.join(directory, filename))
		plt.close(fig)
		im = Image(os.path.join(directory, filename), width*inch, height*inch)
		story.append(im)
		story.append(Spacer(.125, 12))

	
	story.append(Paragraph(title, style=styles["Heading1"]))
	print("plotting global control separation")
	story.append(Paragraph("Control Separation", style=styles["Heading2"]))
	story.append(Paragraph("Control Histogram", style=styles["Heading3"]))
	paragraph = Paragraph(
		"A direct visualization of control separation."
	)
	story.append(paragraph)
	fig = qc_compare_plot(control_histogram, data, "gene_effect", metrics,  
		plot_width, plot_height,
						positive_control_genes=positive_control_genes,
						negative_control_genes=negative_control_genes)
	add_image("control_histogram.png")


	story.append(Paragraph("Per Model QC Metrics", style=styles["Heading3"]))
	print("plotting per-screen control separation")
	story.append(Paragraph(
		"Head-to-head comparison of control separation for each model (cell line).\
For NNMD, more negative is better. For AUROC, more positive is better."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	nnmds = {key: v['gene_effect'].apply(lambda x: 
									nnmd(x.reindex(positive_control_genes), x.reindex(negative_control_genes)),
										axis=1) 
						for key, v in data.items()}
	density_scatter(nnmds[keys[0]], nnmds[keys[1]], diagonal=True, label_outliers=4, s=10, alpha=.5)
	plt.title("NNMD")
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	plt.sca(axes[1])
	aurocs = {key: v['gene_effect'].apply(lambda x: 
									auroc(x.reindex(positive_control_genes), x.reindex(negative_control_genes)), 
										axis=1) 
						for key, v in data.items()}
	density_scatter(aurocs[keys[0]], aurocs[keys[1]], diagonal=True, label_outliers=4, s=10, alpha=.5)
	plt.title("ROC AUC")
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	add_image("model_qc_comparison.png")

	print("plotting selective dependency separation")
	header = Paragraph("Selective Dependency Distinction", style=styles["Heading3"])
	story.append(header)
	paragraph = Paragraph(
		"For known cancer dependencies, the gene effect score with vs without the known indication.\
Ideally each point would fall inthe bottom right corner."
	)
	story.append(paragraph)
	fig = qc_compare_plot(selective_mutated_vs_not_scatter, data, "gene_effect", metrics,
		plot_width, plot_height,  
						mutation_matrix=mutation_matrix)
	add_image("selective_dependencies.png")
	print("plotting expression addictions")
	fig = qc_compare_plot(expression_addiction_volcano, data, "gene_effect", metrics, 
		plot_width, plot_height, 
						addiction_expressions=addiction_expressions)
	add_image("expression_addiction.png")
	
	print("plotting gene differences between datasets")
	story.append(Paragraph("Key Differences", style=styles["Heading2"]))
	story.append(Paragraph(
		"The correlation of individual genes between datasets, both vs mean effect \
and vs the difference of means between \
the two datasets. Below is the direct comparison of gene means in each dataset \
and a comparison of the most extreme values for each gene's score."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	plt.sca(axes[0])
	gene_corr_vs_mean(data[keys[0]]["gene_effect"], data[keys[1]]['gene_effect'],
					metrics=metrics["joint"])
	plt.sca(axes[1])
	gene_corr_vs_mean_diff(data[keys[0]]["gene_effect"], data[keys[1]]['gene_effect'],
					metrics=metrics["joint"])
	plt.xlabel("%s Mean - %s Mean" % tuple(keys))
	add_image("gene_corrs.png")

	fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_width - 2))
	plt.sca(ax)
	density_scatter(data[keys[0]]['gene_effect'].mean(), data[keys[1]]['gene_effect'].mean(), diagonal=True, 
					label_outliers=10, alpha=.5, s=10)
	plt.title("Mean Gene Effect")
	plt.xlabel(keys[0])
	plt.ylabel(keys[1])
	add_image("gene_means.png")
	fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_width - 2))
	plt.sca(ax)
	gene_outlier_plot(data[keys[0]]['gene_effect'], data[keys[1]]['gene_effect'], metrics=metrics['joint'])
	plt.title("Most Extreme Z-Scores by Gene")
	plt.xlabel(keys[0] + " Extreme ZScore")
	plt.ylabel(keys[1] + " Extreme ZScore")
	add_image("gene_zscore_extremes.png")
	story.append(PageBreak())

	story.append(Paragraph("Library Integration", style=styles['Heading2']))

	print("plotting library UMAPs")		 
	story.append(Paragraph("Library Integration UMAP", style=styles["Heading3"]))
	story.append(Paragraph(
		"Embedding of models in gene effect space colored by library coverage."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	for i, key, in enumerate(keys):
		plt.sca(axes[i])
		plt.title(key)
		check_integration_umap(data[key]["gene_effect"], data[key]['sequence_map'], metrics=metrics[key],
								)
	add_image("integration_umap.png")

	print("plotting library mean deviation")
	story.append(Paragraph("Library Mean Deviation", style=styles["Heading3"]))
	story.append(Paragraph(
		"How far a gene's average within a library deviates from the overall average, by library. \
Here, a lowess trend is fitted per library to the squared difference of the gene's mean within \
models screened with the library and its mean overall. Note that the two plots are not necessarily \
on the same scale."
	))
	fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
	for i, key, in enumerate(data.keys()):
		plt.sca(axes[i])
		plt.title(key)
		check_integration_mean_deviation(data[key]["gene_effect"], data[key]['sequence_map'], metrics=metrics[key],
								)
	add_image("integration_deviation.png")
	story.append(PageBreak())
	
	print("plotting genes with low agreement")
	story.append(Paragraph("Exploring Low Agreement Genes", style=styles['Heading2']))
	story.append(Spacer(.125, 12))
	story.append(Paragraph("In the remaining plots, the genes with lowest agreement are explored further. \
NA results for guide efficacy are replaced with -.1"))
	story.append(Spacer(.125, 12))
	lfc = {}
	guide_map = {}
	for key in keys:
		for library in data[key]['logfoldchange']:
			if not library in lfc:
				lfc[library] = data[key]['logfoldchange'][library]
				guide_map[library] = data[key]['guide_map'][library]
			else:
				aligned_left, aligned_right = lfc[library].align(data[key]['logfoldchange'][library],
																join='outer')
				lfc[library] = aligned_left.mask(aligned_left.isnull(), aligned_right)
				guide_map[library] = pd.concat(
					[guide_map[library],  data[key]['guide_map'][library]],
					ignore_index=True
				).drop_duplicates(subset=['sgrna', 'gene'])
	outliers = set(metrics['joint']['worst_agreement']) \
				| set([s.split('_')[0] for s in metrics['joint']['low_outliers']]) \
				| set([s.split('_')[0] for s in metrics['joint']['high_outliers']])
	for gene in outliers:
		print("\t%s" % gene)
		header = Paragraph(gene, style=styles["Heading3"])
		story.append(header)
		fig = interrogate_gene_compare(data, lfc, guide_map, gene, plot_width, plot_width)
		add_image(gene + '.png')
		story.append(PageBreak())


	print("building report")
	doc.build(story)
	return metrics