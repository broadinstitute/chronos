# Chronos: an algorithm for inferring gene fitness effects from CRISPR knockout experiments. 

Copyright 2021 Joshua M. Dempster

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

A full description and benchmarking of the algorithm are available in a preprint: https://doi.org/10.1101/2021.02.25.432728

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

Chronos is competitive with or superior to the other CRISPR algorithms we tested given readcounts from only one late time point, but it will perform even better with multiple late time points if your experiment has them.

You can also use several Chronos tools independently of running the full model. The most relevant is `alternate_CN`, a copy number correction method that accepts any gene effect matrix and a gene-level copy number matrix and returns a corrected gene effect matrix. Additionally, `calculate_fold_change` will convert a readcounts matrix into a fold change in relative abundance using the same procedure as DepMap uses for calculating the Achilles logfold change file. `nan_outgrowths` will remove readcounts suspected to be caused by clonal outgrowth (see Michlits et. al., https://doi.org/10.1038/nmeth.4466 for a description of this phenomenon in CRISPR screens). Finally, `read_hdf5` and `write_hdf5` are useful for efficiently and quickly reading and writing large matrices (as pandas `DataFrames`).


# Installation
**IF YOU ARE USING A MAC WITH AN M1 PROCESSESOR**
Pip's installation of tensorflow WILL NOT work (as of: 06/07/2023). Here is a suggested procedure to get tensorflow installed on M1 machines: https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706. Make sure you can successfully import tensorflow in python before installling Chronos: from the console run `python` then `import tensorflow` and verfy that the kernel doesn't crash. 

Download this repository, navigate to its directory, and in a console run

`    $ python setup.py install`

or 

`    $ pip install .`

in a terminal window. Chronos requires `python 3` with the packages `tensorflow 1.15`, `numpy`, `pandas`, `patsy`, and `h5py`. 

# Getting Started
If you have jupyter notebook, you should run through `Vignette.ipynb`. This will both verify that you have a working installation and demonstrate a typical workflow for Chronos. Chronos is meant to be run in a python environment. 

To run Chronos, you need three Pandas dataframes:

1. A matrix of raw readcounts, where the columns are targeting sgRNAs, the rows are pDNA sequencing samples or replicate samples, and the entries are the number of reads of the given sgRNA in the given sample. Notice that in Chronos matrices, GUIDES and GENES are always COLUMNS and SAMPLES are always ROWS. Readcounts can have null values as long as no column or row is entirely null.

2. A table with at least two columns, `sgrna` and `gene`, mapping the sgRNAs to genes. Chronos will not accept sgRNAs that map to more than one gene. This is intentional. `sgrna` entries should match the columns in raw readcounts. `gene` can be in any format.

3. A table with at least four columns, `sequence_ID`, `cell_line_name`, `pDNA_batch`, and `days`, mapping sequencing samples to cell lines and pDNA measurements. `sequence_ID` should match the row names of the raw readcounts. `days` is the number of days between infection and when the sample was collected, should be integer or float. It will be ignored for pDNA samples. `cell_line_name` MUST be "pDNA" for pDNA samples. if, instead of pDNA, you are sequencing your cells at a very early time point to get initial library abundance, treat these as pDNA samples. If you don't have either, Chronos may not be the right algorithm for your experiment. `pDNA_batch` is needed when your experiment combines samples that have different pDNA references (within the same library). This is the case for Achilles because the PCR primer strategy has changed several times during the course of the experiment. pDNA samples belonging to the same batch will be combined into a single reference. If you don't have pDNA batches, just fill this column some value, such as "batch1".

We've found that a small number of clones in CRISPR cell lines will exhibit dramatic outgrowth that seems unrelated to the intended CRISPR perturbation. We recommend you remove these in place by running

	import chronos
	chronos.nan_outgrowths(readcounts, sequence_map, guide_gene_map)

You can then initialize the Chronos model

    model = chronos.Chronos(
    	readcounts={'my_library': readcounts},
    	sequence_map={'my_library': sequence_map},
    	guide_gene_map={'my_library': guide_gene_map}
    )

This odd syntax is used because it allows you to process results from different libraries at the same time. If you have libraries 1 and 2, and readcounts, sequence maps, and guide maps for them, you would initialize Chronos as such:

    model = chronos.Chronos(
    	readcounts={'my_library1': readcounts1, 'my_library2': readcounts2},
    	sequence_map={'my_library': sequence_map, 'my_library2': sequence_map2},
    	guide_gene_map={'my_library': guide_gene_map, 'my_library2': guide_gene_map2}
    )

Either way, you can then train Chronos by calling 

    model.train()

Once the model is trained, you can save all the parameters of interest by calling 

    model.save("my_save_directory")

You can also directly access model parameters, for example:

	gene_effect = model.gene_effect
	guide_efficacy = model.guide_efficacy

If you have labeled gene_level copy number data, Chronos has an option to correct the gene effect matrix. We recommend first globally normalizing the gene effect matrix so the median of all common essential gene scores is -1 and the median of all nonessential genes is 0. Unlike CERES outputs, we do NOT recommend normalizing per cell line. Chronos includes parameters like `cell_line_growth_rate` and `cell_line_efficacy` along with other regularization terms that help align data between cell lines. 

    gene_effect -= gene_effect.reindex(columns=my_nonessential_gene_list).median(axix=1).median()
    gene_effect /= gene_effect.reindex(columns=my_essential_gene_list).median(axis=1).abs().median()
    gene_effect_corrected, shifts = chronos.alternate_cn(gene_effect, copy_number)
    chronos.write_hdf5(gene_effect_corrected, "my_save_directory/gene_effect.hdf5")

The copy number matrix needs to be aligned to the gene_effect_matrix. Additionally, we assume that it is in the current CCLE format: log2(relative CN + 1), where CN 1 means the relative CN matches the reference. This may still work fine with CN with different units, but has not been tested. 

# Expected run times
The full Achilles dataset takes 3-4 hours to run a gcloud VM with 52 GB of memory. Training the vignette in this package should take around 2 minutes on a typical laptop.

# Other Chronos Options
The Chronos model has a large number of hyperparameters which are described in the model code. Generally we advise against changing these. We've tested them in a wide variety of experimental settings and found the defaults work well. However, a few may be worth tweaking if you want to try and maximize performance. If you do choose to tune the hyperparameters, make sure you evaluate the results with a metric that captures what you really want to get out of the data. We decribe those that might be worth changing.

- `gene_effect_hierarchical` and `gene_effect_smoothing`: The first of these is a CERES style penalty that punishes gene effect scores in individual cell lines for deviating from the mean. The second punishes the deviation of a REGION of gene effect scores in a cell line from the mean, where a region is a contiguous block of genes arranged by their mean gene effect. Cranking up the first of these will reduce the variance within genes, potentially losing interesting differences between samples (but improving measures of control separation within samples). Cranking up the second can produce artifacts in the tails of gene effect, especially if `gene_effect_hierarchical` is too low. If you don't care about differences between samples, or have strong reason to believe all your samples should give the same results, you could consider increasing both of these. 

- `kernel_width`: this is the width of the gaussian kernel applied for `gene_effect_smoothing`. The number of genes used to calculation regional deviation from the mean for each gene will be 6x this number, 3x in each direction from the gene in question. Consider reducing this from its default value (5) for subgenome libraries.

- `cell_efficacy_guide_quantile`: Chronos pre-estimates how efficacious a cell line is (you could think of this as related to Cas9 activity in the cell line). To do this, it looks at the nth percentile guide's log fold change and takes that as the maximum real depletion the cell line can achieve. If screening a small library, especially one highly biased towards essentials, you might consider increasing it from the default value of 0.01. 

- `excess_variance`: how much more noise your readcounts have than a negative binomial model would expect. You can infer this per cell line using the noise between replicates (we don't recommend edgeR for this purpose; we had bad results) and pass that in lieu of a constant value. Alternatively, if you suspect your screen is noisy and you observe that Chronos seems to be assigning high efficacy to the less depleted guides, you can try increasing this value up to 1.

- `scale_cost`: amplifies or diminishes the cost function. Lowering this value effectively increases the strength of all regularization terms. 
	
