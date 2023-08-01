import os
import json
import requests
import pandas as pd
from .model import write_hdf5


def fetch_parameters(url_loc="Data/DepMapDataURLs.json", 
	output_dir="Data/DepMapParameters/", overwrite=False,
	relative_to_chronos=True
):
	'''
	Fetch a set of trained Chronos parameters located at the urls in 
	the json file at `url_loc` (see default file for an example)
	and writes them to the local directory in `output_dir`. 
	Files present will be skipped unless `overwrite` is `True`.
	Both `url_loc` and `output_dir` are relative to the chronos package
	unless `relative_to_chronos` is `False`.
	'''
	chronos_dir = os.path.dirname(__file__)
	if not url_loc.startswith("/"):
		if relative_to_chronos:
			print("`url_loc` will be found relative to the chronos package directory\n'%s'\n\
Pass `relative_to_chronos=False` to make the path relative to your current working directory\n'%s'\n\
instead.\n" % (chronos_dir, os.getcwd()))
			url_loc = os.path.join(chronos_dir, '..', url_loc)

	if not output_dir.startswith("/"):
		if relative_to_chronos:
			print("`output_dir` will be found relative to the chronos package directory\n'%s'\n\
Pass `relative_to_chronos=False` to make the path relative to your current working directory\n'%s'\n\
instead.\n" % (chronos_dir, os.getcwd()))
			output_dir = os.path.join(chronos_dir, '..', output_dir)
			if not os.path.isdir(output_dir):
				os.mkdir(output_dir)

	print("downloading files to %s" % output_dir)

	url_dict = json.loads(open(url_loc).read())

	for filename, url in url_dict.items():
		path = os.path.join(output_dir, filename)
		if filename in os.listdir(output_dir) and not overwrite:
			print("Skipping %s as it already exists, pass `overwrite=True` to overwrite" % filename)
		else:
			print("fetching %s from %s" % (filename, url))
			file = requests.get(url, allow_redirects=True)
			open(path, 'wb').write(file.content)
	print("all files fetched, tranforming format")
	reformat_directory(output_dir)
	print('done')



def reformat_directory(directory):
	'''transforms file formats in `directory` from DepMap release format to Chronos' expected format for `import_model`'''
	if not "gene_effect.hdf5" in os.listdir(directory):
		print("transforming gene_effect.csv, this may take a minute")
		ge = pd.read_csv(os.path.join(directory, "gene_effect.csv"), index_col=0)
		write_hdf5(ge, os.path.join(directory, "gene_effect.hdf5"))

	print("transforming guide efficacy")
	guide_eff = pd.read_csv(os.path.join(directory, "guide_efficacy.csv"))
	guide_eff.rename(columns={"sgRNA": "sgrna", "Efficacy": "efficacy"}, errors="ignore", inplace=True)
	guide_eff.to_csv(os.path.join(directory, "guide_efficacy.csv"), index=None)


