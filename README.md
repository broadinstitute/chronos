# Chronos: an algorithm for inferring gene fitness effects from CRISPR knockout experiments. 

A full description and benchmarking of Chronos 1 are available in a publication: https://doi.org/10.1186/s13059-021-02540-7

An additional preprint describing the changes made to Chronos 2 will be released when the underlying data is public, expected 2024.

# When to use it
Chronos is well suited for any CRISPR KO experiment where:
- You measured initial pDNA sgRNA readcounts and readcounts at one or more later time points.
- You might have one or more cell lines.
- You might have one library, or be combining data from multiple libraries.
- Genome-wide or sub-genome coverage.
- You expect most cells to be proliferating.
- You expect the majority of gene knockouts to have little to no effect on proliferation.
- You might or might not have copy number data for your cell lines.

Chronos may not work well for:
- RNAi experiments. Chronos makes biological assumptions that are fundamentally incompatible with RNAi. Try DEMETER 2.
- Rescue experiments. If most cells are dying, we can't offer any guarantees of Chronos' performance.
- A focused essential gene library, for the same reason. 
- Multi-condition experiments where your only control is a late time point (such as DMSO). Chronos requires pDNA abundance.

We strongly recommend having at least two sgRNAs per gene. This is true regardless of the algorithm you use.

Chronos is competitive with or superior to the other CRISPR algorithms we tested given readcounts from only one late time point, but it will perform even better with multiple late time points if your experiment has them.


# Installation

## Note on Mac M1 chips

As of 2023/07/14, `pip` will not correctly install tensorflow on Macs with the M1 chip. You can try the instructions here: https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706 but they appear to fail for python 3.10. Reversion to 3.8 led to success. Make sure you can import tensorflow in python without the kernel hanging before proceeding. 

If you followed those instructions, you may find that running the vignette with multiple libraries could also cause the error "cannot assign a device for operation ReadVariableOp." Uninstalling tensorflow-metal will fix this error but may disable running on the GPU. But in general we do not recommend running Chronos on the GPU.

## Installing Chronos

If you have `pip` installed, you can install Chronos from pyPI with

`    $ pip install crispr_chronos`

However, we recommend downloading this repository as well to run the vignette and download the DepMap trained Chronos parameters.

Chronos `model` requires `python 3` with the packages `tensorflow 2.x`, `numpy`, `pandas`,`h5py`. However, additional modules require additional packages which will be installed by default if missing: `patsy`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `adjust_text`, `scikit-learn`, `umap`, `reportlab`.

# Getting Started
If you have jupyter notebook, you should run through `Vignette.ipynb`. This will both verify that you have a working installation and demonstrate a typical workflow for Chronos. Chronos is meant to be run in a python environment. 

To run Chronos, you need a minimum of three Pandas dataframes:

1. A matrix of raw readcounts, where the columns are targeting sgRNAs, the rows are pDNA sequencing samples or replicate samples, and the entries are the number of reads of the given sgRNA in the given sample. Notice that in Chronos matrices, GUIDES and GENES are always COLUMNS and SAMPLES are always ROWS. Readcounts can have null values as long as no column or row is entirely null.

2. A table with at least two columns, `sgrna` and `gene`, mapping the sgRNAs to genes. Chronos will not accept sgRNAs that map to more than one gene. This is intentional. `sgrna` entries should match the columns in raw readcounts. `gene` can be in any format.

3. A table with at least four columns, `sequence_ID`, `cell_line_name`, `pDNA_batch`, and `days`, mapping sequencing samples to cell lines and pDNA measurements. `sequence_ID` should match the row names of the raw readcounts. `days` is the number of days between infection and when the sample was collected, should be integer or float. It will be ignored for pDNA samples. `cell_line_name` MUST be "pDNA" for pDNA samples. if, instead of pDNA, you are sequencing your cells at a very early time point to get initial library abundance, treat these as pDNA samples. If you don't have either, Chronos may not be the right algorithm for your experiment. `pDNA_batch` is needed when your experiment combines samples that have different pDNA references (within the same library). This is the case for Achilles because the PCR primer strategy has changed several times during the course of the experiment. pDNA samples belonging to the same batch will be combined into a single reference. If you don't have pDNA batches, just fill this column some value, such as "batch1".

To benefit from improved normalization and allow Chronos to infer the overdispersion of screens, supplying a list or array of `negative_control_sgrnas` is also necessary. These are simply the sgRNAs which you believe should have no viability effect in any of your screens. It is much better to use cutting than noncutting controls, and as many as possible.

We've found that a small number of clones in CRISPR cell lines will exhibit dramatic outgrowth that seems unrelated to the intended CRISPR perturbation. We recommend you remove these in place by running

	import chronos
	chronos.nan_outgrowths(readcounts, sequence_map, guide_gene_map)

You can then initialize the Chronos model

    model = chronos.Chronos(
    	readcounts={'my_library': readcounts},
    	sequence_map={'my_library': sequence_map},
    	guide_gene_map={'my_library': guide_gene_map},
        negative_control_sgrnas={'my_library': negative_control_sgrnas}
    )

This odd syntax is used because it allows you to process results from different libraries at the same time. If you have libraries 1 and 2, and readcounts, sequence maps, guide maps, and negative control sgRNAs for them, you would initialize Chronos as such:

    model = chronos.Chronos(
    	readcounts={'my_library1': readcounts1, 'my_library2': readcounts2},
    	sequence_map={'my_library': sequence_map, 'my_library2': sequence_map2},
    	guide_gene_map={'my_library': guide_gene_map, 'my_library2': guide_gene_map2},
        negative_control_sgrnas={'my_library1': negative_control_sgrnas1, 'my_library2': negative_control_sgrnas2}
    )

Either way, you can then train Chronos by calling 

    model.train()

Once the model is trained, you can save all the parameters by calling

    model.save("my_save_directory")

You can also directly access model parameters, for example:

	gene_effect = model.gene_effect
	guide_efficacy = model.guide_efficacy

`gene_effect` is the primary attribute you will be interested in in 99% of use cases. It is a numerical matrix indexed on rows by `cell_line_name` and on columns by `gene`, with values indicating the _relative change in growth rate_ caused by successful knockout of the gene. 0 indicates no change, negative values a loss of viability, and positive values a gain of viability. NaNs in this matrix can occur because no sgRNAs targeting the gene

Note some parameters will be dictionaries or tables, because they are learned separately per library. 

If you have labeled gene_level copy number data, Chronos has an option to correct the gene effect matrix. We recommend first globally normalizing the gene effect matrix so the median of all common essential gene scores is -1 and the median of all nonessential genes is 0. Unlike CERES outputs, we do NOT recommend normalizing per cell line. Chronos includes parameters like `cell_line_growth_rate` and `cell_line_efficacy` along with other regularization terms that help align data between cell lines. 

    gene_effect -= gene_effect.reindex(columns=my_nonessential_gene_list).median(axix=1).median()
    gene_effect /= gene_effect.reindex(columns=my_essential_gene_list).median(axis=1).abs().median()
    gene_effect_corrected, shifts = chronos.alternate_cn(gene_effect, copy_number)
    chronos.write_hdf5(gene_effect_corrected, "my_save_directory/gene_effect.hdf5")

The copy number matrix needs to be aligned to the gene_effect_matrix. Additionally, we assume that it is in the current CCLE format: log2(relative CN + 1), where CN 1 means the relative CN matches the reference. This may still work fine with CN with different units, but has not been tested. 

New functionality in Chronos 2.x includes two types of quality control reports, one you can run on your raw data, the other on the trained Chronos results, and the ability to load DepMap public Chronos runs and use the trained parameters for processing your own screens (if they are in a public DepMap library, currently just Avana and KY). See the vignette for details on how to do this.

# Expected run times
The full Achilles dataset takes 3-4 hours to run a gcloud VM with 52 GB of memory. Training the vignette in this package should take around 2 minutes on a typical laptop.

# Other Chronos Options
The Chronos model has a large number of hyperparameters which are described in the model code. Generally we advise against changing these. We've tested them in a wide variety of experimental settings and found the defaults work well. However, a few may be worth tweaking if you want to try and maximize performance. If you do choose to tune the hyperparameters, make sure you evaluate the results with a metric that captures what you really want to get out of the data. We decribe the hyperparameters that might be worth changing here.

- `gene_effect_hierarchical` and `gene_effect_smoothing`: The first of these is a CERES style penalty that punishes gene effect scores in individual cell lines for deviating from the mean. The second punishes the deviation of a REGION of gene effect scores in a cell line from the mean, where a region is a contiguous block of genes arranged by their mean gene effect. Cranking up the first of these will reduce the variance within genes, potentially losing interesting differences between samples (but improving measures of control separation within samples). Cranking up the second can produce artifacts in the tails of gene effect, especially if `gene_effect_hierarchical` is too low. If you don't care about differences between samples, or have strong reason to believe all your samples should give the same results, you could consider increasing both of these. 

- `kernel_width`: this is the width of the gaussian kernel applied for `gene_effect_smoothing`. The number of genes used to calculation regional deviation from the mean for each gene will be 6x this number, 3x in each direction from the gene in question. Consider reducing this from its default value (50) for subgenome libraries.

- `cell_efficacy_guide_quantile`: Chronos pre-estimates how efficacious a cell line is (you could think of this as related to Cas9 activity in the cell line). To do this, it looks at the nth percentile guide's log fold change and takes that as the maximum real depletion the cell line can achieve. If screening a small library, especially one highly biased towards essentials, you might consider increasing it from the default value of 0.01. 

- `library_batch_reg`: this regularizes the mean gene effect within libraries towards the mean effect across libraries. Has no effect unless you have more than one library in the run. Note that this is one of two Chronos properties that removes library batch effects; the other is the internal matrix of `library_batch_effect`, which can't be turned off. If you think there should be real biological differences between your libraries, consider concatenating the input files into a single pseudolibrary. On the other hand, if you have two screen batches in the same library and you want to correct batch effects, you can split your screens into two pseudolibraries with the same sgRNAs in each.

- `scale_cost`: amplifies or diminishes the cost function. Lowering this value effectively increases the strength of all regularization terms. 


# Tools that are useful outside of Chronos:

## Preprocessing tools:

- `nan_outgrowths` will remove readcounts suspected to be caused by clonal outgrowth (see Michlits et. al., https://doi.org/10.1038/nmeth.4466 for a description of this phenomenon in CRISPR screens). 

- `normalize_readcounts` will sum pDNA measurements of the same pDNA batch, align the different batches by mode in log space, then align replicates to their pDNA batch by median abundance of the negative controls (if negative controls are supplied)

- `calculate_fold_change` will convert a readcounts matrix into a fold change. Will use RPM normalization by default, which will undo the normalization in `normalize_readcounts`

- `estimate_alpha` estimates the overdispersion parameter of the NB2 negative binomial counts model on a per-replicate basis using negative controls


## Postprocessing tools:

- `alternate_CN`, a copy number correction method that accepts any gene effect matrix and a gene-level copy number matrix and returns a corrected gene effect matrix. 


## QC reports (requires the matplotlib, seaborn, and reportlab packages):

- `reports.qc_initial_data` takes in readcounts, a guide map, a sequence map, and optionally postive and negative control sgRNAs and provides a number of plots and metrics to assess the quality of CRISPR screen data.

- `reports.qc_dataset` evaluates data quality after Chronos processing. You will want to call `.save` on your trained model to create a properly formatted directory to load with this function. Some aspects of the QC require omics data in various forms. See the vignette for a walkthrough.


## Generally useful functions:

- `read_hdf5` and `write_hdf5` allow you to translate numerical matrices between pandas DataFrames and effiicient binary files.

- `evaluations.fast_cor` efficiently computes the correlation matrix of one or two matrices (pandas DataFrames) with block null values. `evalautions.fast_cor_core` accepts numpy arrays as inputs instead.

- `evaluations.nnmd`, `evaluations.auroc`, and `evaluations.pr_auc` compute control separation metrics

- `plotting.density_scatter` produces a scatter plot with points colored by density, a trendline (much more efficient than seaborn's version), and optionally a diagonal, along with several options for labeling outlier points.

- `plotting.binplot` turns scatter data into a boxplot by binning one axis, which can reveal trends that are hard to see with scatter

- `plotting.dict_plot` takes a dictionary of data and produces a subplot per entry, titled with its key.
	
