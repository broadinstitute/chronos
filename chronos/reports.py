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

from matplotlib import rcParams
from matplotlib import cycler as color_cycler

from copy import copy

okabe_ito = [
    [230/255, 159/255, 0],
    [86/255, 180/255, 233/255],
    [0, 158/255, 115/255],
    [204/255, 121/255, 167/255],
    [0, 114/255, 178/255],
    [213/255, 94/255, 0],
    [240/255, 228/255, 66/255]
]

matplotlib_rcParams_update = {
	'axes.titlesize': 11,
	'axes.spines.right': False,
	'axes.spines.top': False,
	'savefig.dpi': 200,
	'savefig.transparent': False,
	'font.family': 'Arial',
	'font.size': '10',
	'figure.dpi': 200,
	"savefig.facecolor": (1, 1, 1.0, 0.2),
	'xtick.labelsize': 9,
	'ytick.labelsize': 9,
	'legend.fontsize': 7,
	'axes.prop_cycle': color_cycler(color=okabe_ito),
}

#set default cycle to Okabe-Ito




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
		'replicate_efficacy': pd.read_csv(os.path.join(directory, "replicate_efficacy.csv"), index_col=0),
		'growth_rate': pd.read_csv(os.path.join(directory, "growth_rate.csv"), index_col=0),
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

def sum_collapse_dataframes(dfs):
	'''
	Given an iterable of pandas DataFrames, returns a single dataframe
	where each value is given by the sum of values for the same index/column
	across the input DataFrames, filling NaNs with 0.
	'''
	numerator = None
	for df in dfs:
		if numerator is None:
			numerator = df.fillna(0)
		else:
			numerator, df = numerator.align(df, join='outer')
			numerator.fillna(0, inplace=True)
			numerator += df.fillna(0).values
	return numerator
	

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
		  specific_plot_dimensions={}, report_worst_lines=False
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
		`report_worst_lines` (`bool`): whether to generate detailed plots for the cell lines with worst performance
	Returns:
		`dict` containing the calculated QC metrics, which will also be in the report.
	'''

	original_rcParams = copy(rcParams)
	rcParams.update(matplotlib_rcParams_update)

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
		reps = sequence_map.query("cell_line_name == @line").sequence_ID
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
	
	if not report_worst_lines:
		doc.build(story)
		rcParams.update(original_rcParams)
		return metrics

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

	for cell_line in worst:
		story.append(PageBreak())
		story.append(Paragraph(line, style=styles["Heading3"]))

		reps = sequence_map.query("cell_line_name == @cell_line").sequence_ID.unique()
		rep_labels = dict(zip(reps, trim_overlapping_lead_and_tail(reps)))
		n = 0
		titles = {}
		for i in range(len(reps)-1):
			for j in range(i+1, len(reps)):
				n += 1
				titles["%s %i" % (cell_line, n)] = (reps[i], reps[j])

		for i in range(0, len(titles)+len(titles)//2, 2):
			fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))

			for j in range(2):
				try:
					title = list(titles.keys())[i+j]
				except IndexError:
					fig.delaxes(axes[j])
					continue
				plt.sca(axes[j])
				replicate_plot(normalized, *titles[title])
				plt.xlabel("Rep." + rep_labels[titles[title][0]])
				plt.ylabel("Rep." + rep_labels[titles[title][1]])
				plt.title(title)
			plt.tight_layout()

			add_image(f"replicate_corr_plots_{line}_{i}.png")


		paired_pDNA_plots(normalized, sequence_map, line, negative_control_sgrnas, positive_control_sgrnas,
						 plot_width, plot_height)
		add_image("%s_pdna_plot.png" % line)
		
	doc.build(story)
	
	rcParams.update(original_rcParams)
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
						 "excess_variance", "growth_rate", "replicate_efficacy",
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

	original_rcParams = copy(rcParams)
	rcParams.update(matplotlib_rcParams_update)

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

	if len(data["gene_effect"]) > 1:
		story.append(Paragraph("Statistical Properties of Gene Effects", style=styles["Heading3"]))
		print("plotting gene effect mean relationships")

		story.append(Paragraph(
	"Higher overall gene SD is better (if control separation in each cell line is maintained). There is usually a trend \
	towards more variance in more negative genes. There should NOT be a trend in the copy number plot."
	))
		fig, axes = plt.subplots(1, 1, figsize=(plot_width, plot_height))
		mean_vs_sd_scatter(data["gene_effect"], metrics=metrics)

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
		replicate_efficacy = []

		for library in library_data:

			gr, cle = data["growth_rate"].query("library == @library")["growth_rate"].dropna().align(
				data['replicate_efficacy'].query("library == @library")["replicate_efficacy"].dropna(), 
				join="inner"
			)

			growth_rate.append(gr)
			replicate_efficacy.append(cle)

		growth_rate, replicate_efficacy = pd.concat(growth_rate), pd.concat(replicate_efficacy)
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))
		plt.sca(axes[0])
		density_scatter(growth_rate, replicate_efficacy, trend_line=False, outliers_from="xy_zscore")
		plt.xlabel("Relative Growth Rate")
		plt.ylabel("Replicate Screening Efficacy")
		metrics["growth_rate_sd"] = growth_rate.std()
		metrics["cell_efficacy_mean"] = replicate_efficacy.mean()
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
			check_integration_mean_deviation(data['gene_effect'], data['sequence_map'], data["guide_map"], metrics=metrics)
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

	if len(data["gene_effect"]) > 1:
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

	if len(data["gene_effect"]) > 1:
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

	if len(data["gene_effect"]) > 1:
		story.insert(1, Paragraph(
'''
Summary: the standard deviation (SD) of gene means in gene effect is %1.3f.\n
The mean of gene SDs is %1.3f the SD of gene means.\n
The SD of cell line means is %1.3f the SD of gene means\n. 
The correlation of each library's mean LFC per gene with Chronos' mean gene effect is:\n
%s
''' % (ge_mean.std(), metrics['mean_SD:SD_means'], cell_line_mean, naive_corr_text)
		))
			

	print("building report")
	doc.build(story)

	rcParams.update(original_rcParams)
	
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
		check_integration_mean_deviation(data[key]["gene_effect"], data[key]['sequence_map'], data[key]["guide_map"],
			 metrics=metrics[key],
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


def hit_calling_report(
	title,
	report_name=None, 
	directory='.', 
	gene_effect_file="gene_effect.hdf5",
	p_value_file="p_value.hdf5",
	frequentist_fdr_file="frequentist_fdr.hdf5",
	probability_file="probability_dependent.hdf5",
	bayesian_fdr_file="bayesian_fdr.hdf5",
	full_gene_effect_file=None,
	plot_width=7.5, plot_height=3.25,
	doc_args=dict(
		pagesize=letter, rightMargin=.5*inch, leftMargin=.5*inch,
		topMargin=.5*inch,bottomMargin=.5*inch
	),
	specific_plot_dimensions={}
):
	'''
	Report summarizing the hits and biology discovered in the Chronos run
	Parameters:
		`title` (`str`): the report title, printed on first page
		`report_name` (`str`): an optional file name for the report. If none is provided, `title` + '.pdf' will be used.
		`directory` (`str`): where the report and figure panels will be generated.
		`gene_effect_file` (`str`): path to hdf5 file containing desired gene effects
		`p_value_file` (`str`): path to hdf5 file containing p_value estimates
		`frequentist_fdr_file` (`str`): path to hdf5 file containing FDR estimates from p-values
		`probability_file` (`str`): path to hdf5 file containing estimated probabilities of dependency
		`bayesian_fdr_file` (`str`): path to hdf5 file containing FDR estimates from the probabilities
		`full_gene_effect_file` (`str`): path to an hdf5 matrix containing gene effects for many cell lines. This is used
			to identify what hits are specific to a given screen vs general. If not provided, `gene_effect_file` will be
			used.
		`plot_width`, `plot_height` (`float`): size of plots that will be put in the report in inches.
		`doc_args` (`dict`): additional arguments will be passed to `SimpleDocTemplate`.
		`specific_plot_dimensions` (`dict` of 2-tuple`): if a plot's name is present, will use the the value
			 to specify dimensions for that plot instead of deriving them from `plot_width` and `plot_height`
	Returns:
		None
	'''

	
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

	original_rcParams = copy(rcParams)
	rcParams.update(matplotlib_rcParams_update)

	print("loading data")
	gene_effect = read_hdf5(gene_effect_file)
	p_value = read_hdf5(p_value_file)
	frequentist_fdr = read_hdf5(frequentist_fdr_file)
	probability = read_hdf5(probability_file)
	bayesian_fdr = read_hdf5(bayesian_fdr_file)

	if full_gene_effect_file is None:
		full_gene_effect = gene_effect
	else:
		full_gene_effect = read_hdf5(full_gene_effect_file)

	story.append(Paragraph(title, style=styles["Heading1"]))

	story.append(Paragraph("False Discovery Rates", style=styles["Heading2"]))
	print("false discovery rates")
	story.append(Paragraph(
		"Discoveries vs gene effect using either frequentist or Bayesian methods. "
		"The Bayesian estimates of false discovery are usually better, but vulnerable "
		"to a bad set of positive controls."
	))
	for line in gene_effect.index:
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))

		plt.sca(axes[0])
		fdr_volcano(gene_effect.loc[line], frequentist_fdr.loc[line])
		plt.ylabel("Frequentist FDR (-log10)")

		plt.sca(axes[1])
		fdr_volcano(gene_effect.loc[line], bayesian_fdr.loc[line])
		plt.ylabel("Bayesian FDR (-log10)")

		add_image("fdr_volcano_%s.png" % line)

	story.append(Paragraph("Specific Biology", style=styles["Heading2"]))
	print("specific biology")
	aligned, full_gene_effect = gene_effect.align(full_gene_effect, axis=1, join="inner")

	zscores = (aligned - full_gene_effect.mean()) / np.sqrt(1 + full_gene_effect.var())

	if len(full_gene_effect) > 1:

		story.append(Paragraph(
			f"For each cell line, the most selective dependencies relative to {len(full_gene_effect)} lines in the "
			f"full gene effect matrix. These are picked by using a regularized z-score (z = (x - mu) / sqrt(1 + sigma^2)) "
			f"and taking hits that have Bayesian FDR < 0.1 and the strongest Z scores. "
			f"This is most useful if the library for these screens is present in the full gene effect matrix "
			f"and a pretrained model was used for Chronos. For each cell line, the top zscored genes are analyzed "
			f"for term enrichment with geneTEA."
		))

		for line in aligned.index:
			fig, axes = plt.subplots(1, 1, figsize=(plot_width, plot_height))
			z = zscores.loc[line]
			z = z[bayesian_fdr.loc[line] < .1]

			candidates = z.sort_values().loc[lambda x: x < -.5].index[:50]
			if len(candidates) == 0:
				print(aligned.loc[line].sort_values())
				print(full_gene_effect)
				continue
			context_box_plot(aligned.loc[line, candidates], full_gene_effect)
			add_image(f"select_dependencies_{line}.png")

			
			plot_enriched_terms([s.split(' ')[0].strip() for s in candidates])
			plt.gcf().set_size_inches(plot_width, plot_height)
			plt.tight_layout()
			add_image(f"select_dependencies_genetea_enrichment_{line}.png")

		


	print("building report")
	doc.build(story)
	rcParams.update(original_rcParams)


def differential_dependency_report(
	title,
	stats_file,
	report_name=None, 
	directory='.', 
	plot_width=7.5, plot_height=3.25,
	doc_args=dict(
		pagesize=letter, rightMargin=.5*inch, leftMargin=.5*inch,
		topMargin=.5*inch,bottomMargin=.5*inch
	),
	specific_plot_dimensions={}
):
	'''
	Report summarizing the hits and biology discovered in the Chronos run
	Parameters:
		`title` (`str`): the report title, printed on first page
		'stats_file' ('str'): the output of `ChronosComparator.compare_conditions`
		`report_name` (`str`): an optional file name for the report. If none is provided, `title` + '.pdf' will be used.
		`directory` (`str`): where the report and figure panels will be generated.
		`plot_width`, `plot_height` (`float`): size of plots that will be put in the report in inches.
		`doc_args` (`dict`): additional arguments will be passed to `SimpleDocTemplate`.
		`specific_plot_dimensions` (`dict` of 2-tuple`): if a plot's name is present, will use the the value
			 to specify dimensions for that plot instead of deriving them from `plot_width` and `plot_height`
	Returns:
		None
	'''

	
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

	original_rcParams = copy(rcParams)
	rcParams.update(matplotlib_rcParams_update)

	print("loading data")
	stats = pd.read_csv(stats_file)
	stats["gene"] = stats.gene.apply(lambda s: s.split(" ")[0])
	conditions = [s.split("gene_effect_in_")[1].strip() for s in stats.columns if "gene_effect_in_" in s]
	stats_by_line = {line: group.set_index("gene") for line, group in stats.groupby("cell_line_name")}

	story.append(Paragraph(title, style=styles["Heading1"]))

	story.append(Paragraph("Differential Dependency", style=styles["Heading2"]))
	print("plots")
	story.append(Paragraph(
		"Which genes are significantly different in the  "
		f"{conditions[0]} condition vs {conditions[1]} condition. "
	))
	for line, table in stats_by_line.items():
		story.append(Paragraph(line, style=styles["Heading3"]))
		fig, axes = plt.subplots(1, 2, figsize=(plot_width, plot_height))

		plt.sca(axes[0])
		fdr = table["likelihood_fdr"]
		ged = table["gene_effect_difference"]
		fdr_volcano(ged, fdr, label_outliers=10, outliers_from="xy_zscore")
		plt.ylabel("FDR (-log10)")
		plt.xlabel(f"Gene Effect Diff. {conditions[1][:12]} - {conditions[0][:12]}")
		plt.title("")

		plt.sca(axes[1])
		density_scatter(table[f"gene_effect_in_{conditions[1]}"], table[f"gene_effect_in_{conditions[0]}"],
			label_outliers=10, diagonal=True)
		plt.xlabel(f"Gene Effect ({conditions[1]})")
		plt.ylabel(f"Gene Effect ({conditions[0]})")
		plt.tight_layout()

		add_image("differential_dependency_%s.png" % line)

		sigup = fdr.index[(fdr < .1) & (ged > 0)]
		if len(sigup) >= 3:
			axes = plot_enriched_terms([s.split(' ')[0].strip() for s in sigup])
			if not (axes is None):
				plt.gcf().suptitle(f"Genes more essential in {conditions[1]}")
				plt.gcf().set_size_inches(plot_width, plot_height)
				plt.tight_layout()
				fig.subplots_adjust(top=0.88)
				add_image(f"diffdep_up_genetea_enrichment_{line}.png")

		sigdown = fdr.index[(fdr < .1) & (ged < 0)]
		if len(sigup) >= 3:
			axes = plot_enriched_terms([s.split(' ')[0].strip() for s in sigdown])
			if not (axes is None):
				plt.gcf().suptitle(f"Genes more essential in {conditions[0]}")
				plt.gcf().set_size_inches(plot_width, plot_height)
				plt.tight_layout()
				fig.subplots_adjust(top=0.88)
				add_image(f"diffdep_down_genetea_enrichment_{line}.png")


	print("building report")
	doc.build(story)
	rcParams.update(original_rcParams)