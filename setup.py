import os
from setuptools import setup, find_packages

extras_require = {
	"copy_correction": ["patsy>=0.5.2"],
	"evaluations": ["matplotlib>=3.6", "seaborn>=0.12", "scikit-learn>=1.1", "statsmodels>=0.13", "scipy>=1.9"],
	"adjust_text": ["adjustText"],
	"embedding": ["umap-learn>=0.5.3"],
	"reports": ["reportlab>=3.6"],
	"model": ["numpy>=1.2", "pandas>=1.3", "tensorflow>2", "h5py>=3.7"],
	"hit_calling": ["scipy>=1.9", "sympy>=1.0", "statsmodels>=0.13"]
}
extras_require['all'] = sorted(set.union(*[set(v) for v in extras_require.values()]))

setup(
	name='crispr_chronos',
	version='2.3.4',
	author="BroadInstitute CDS",
	description="Time series modeling of CRISPR perturbation readcounts in biological data",
	packages=find_packages(),
	package_data={'': ['*.r']},
	install_requires=extras_require['all']
	#extras_require = extras_require
)
