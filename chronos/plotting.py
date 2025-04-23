from __future__ import print_function
import numpy as np
import pandas as pd
from warnings import warn

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

from scipy.interpolate import interpn
from statsmodels.nonparametric.smoothers_lowess import lowess
try:
	from adjustText import adjust_text
	adjust_text_present = True
except:

	adjust_text_present = False


def lowess_trend(x, y, frac=.25, max_points=2000, min_points=50, delta_frac=.01, **kwargs):
	'''
	A wrapper for statsmodel's lowess with a somewhat more useful parameterization
	Parameters:
		`x`, `y`: the points. `y` will be smoothed as a function of `x`.
		`frac`: `float` in [0, 1]. The fraction of the points used for each linear regression.
		`min_points`: `int`. The maximum number of points to be used for each linear regression.
					Overrides `frac` when larger.
		`max_points`: `int`. The maximum number of points to be used for each linear regression.
					Overrides `frac` when smaller.
		`delta_frac`: the fraction of the range of `x` within which to use linear interpolation
		`              instead of a new regression.
		Other args passed to lowess.
	Returns:
		The unsorted smoothed y values.
	'''
	frac = min(max_points/len(x), frac)
	frac = max(frac, min_points/len(x))
	frac = np.clip(frac, 0, 1)
	rng = x.max() - x.min()
	delta = min(delta_frac * rng, 50/len(x)*rng)
	delta = min(delta, rng)
	return lowess(y, x, frac, delta=delta, is_sorted=False, return_sorted=False, **kwargs)


def identify_outliers_by_trend(x, y, n_outliers, y_trend=None, min_outlier_std=3, **kwargs):
	'''
	Get the `n_outliers` farthest from the smoothed trend. y_trend is not supplied, it will be estimated
	using `lowess_trend`.
	Parameters:
		`x`, `y`: the points. `y` will be smoothed as a function of `x`.
		`n_outliers`: how many outliers to return.
		`y_trend`: The trend of y(x). If not provided, will be estimated by lowess.
		`min_outlier_std` (`float`): `y` must be at least this many standard deviations from `y_trend`
				(standard deviation measured as deviation from the trend) to be an outlier.
		Other args are passed to `lowess_trend`, if used.
	Returns:
		the numerical index of the outliers
	'''
	if y_trend is None:
		y_trend = lowess_trend(x, y, **kwargs)

	diff = y-y_trend
	sd = np.std(diff)
	normed = np.abs(diff/sd)
	index = np.arange(len(normed)).astype(int)
	candidates = normed[normed > min_outlier_std]
	index = index[normed > min_outlier_std]
	order = np.argsort(candidates)
	return index[order[-n_outliers:]]
	

def identify_outliers_by_density(x, y, density, n_outliers, candidate_density_quantile=.05, high_density_quantile=.5,
		max_candidates=500, max_high=10000):
	'''
	Identify outliers in 2D space by point density. This is done by first identifying a set of candidate points of lowest
	density, then a set of points with high density, then choosing candidates that have the greatest minimum distance
	to any point with high density.
	Parameters:
		`x`, `y`, `density`: 1D arrays giving the position of each point and the estimated density of points at that position
		`n_outliers`: how many outliers to return. If fewer candidates are found than the number of requested outliers,
				 all candidates will be returned.
		`candidate_density_quantile`: the fraction of points to choose as possible outliers based on density
		`high_density_quantile`: the fraction of points to be treated as dense
	`   max_candidates`: overrides `candidate_density_quantile` if too many candidates are considered. Useful for very large datasets.
	`   max_high`: overrides `high_density_quantile` if too many high density points are considered. Useful for very large datasets.
	Returns:
		the numerical index of the outliers
	'''
	if not (len(x)==len(y)==len(density)):
		raise ValueError("`x`, `y`, and `density` must have the same length")
	if candidate_density_quantile > high_density_quantile:
		raise ValueError("`candidate_density_quantile` must be less than `high_density_quantile`")
	candidate_density_quantile = min(candidate_density_quantile, max_candidates/len(x))
	high_density_quantile = max(high_density_quantile, 1-max_high/len(x))

	candidates = np.arange(len(density)).astype(int)[density < np.quantile(density, candidate_density_quantile)]
	high_density = density > np.quantile(density, high_density_quantile)
	x_diff = np.subtract.outer(x[candidates], x[high_density])
	y_diff = np.subtract.outer(y[candidates], y[high_density])
	r2 = np.square(x_diff) + np.square(y_diff)
	r2_min = r2.min(axis=1)
	farthest = np.argsort(r2_min)[-n_outliers:]
	return candidates[farthest]


def identify_outliers_by_diagonal(x, y, n_outliers):
	'''
	Identify points in 2D space as outliers by distance from the diagonal x==y, i.e. the points with the greatest difference
	between x and y.
	Parameters:
		`x`, `y` : 1D arrays giving the position of each point
		`n_outliers`: how many outliers to return. If fewer candidates are found than the number of requested outliers,
				 all candidates will be returned.
	Returns:
		the numerical index of the outliers
	'''
	diff = np.abs(x - y)
	diff[pd.isnull(diff)] = 0
	order = np.argsort(diff)
	return order[-n_outliers:]


def identify_outliers_by_zscore(x, y, n_outliers):
	'''
	Identify points in 2D space as outliers by zscore. `x` and `y` are first zscored, then combined into a scaled Euclidian distance
	from the mean (`x**2 + y**2`). Those with the greatest distance are returned as outliers.
	Parameters:
		`x`, `y`: 1D arrays giving the position of each point
		`n_outliers`: how many outliers to return. If fewer candidates are found than the number of requested outliers,
				 all candidates will be returned.
	Returns:
		the numerical index of the outliers
	'''
	zx = np.abs(x - np.mean(x))/np.std(x)
	zy = np.abs(y - np.mean(y))/np.std(y)
	r = zx**2 + zy**2
	r[pd.isnull(r)] = 0
	order = np.argsort(r)
	return order[-n_outliers:]


def identify_outliers_1d(x, n_outliers):
	'''
	Identify points in 1D space as outliers by distance from median. 
	Parameters:
		`x`: 1D array
		`n_outliers`: how many outliers to return. If fewer candidates are found than the number of requested outliers,
				 all candidates will be returned.
	Returns:
		the numerical index of the outliers
	'''
	zx = np.abs(x - np.median(x))
	order = np.argsort(zx)
	return order[-n_outliers:]


def get_density(x, y, bins=50):
	'''
	get the 2D density of the 1D arrays `x` and `y` using a histogram with n `bins`
	on each axis
	'''
	try:
		data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
	except ValueError as e:
		print(x)
		print(y)
		print(bins)
		raise e
	z =  interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , 
			data , np.vstack([x,y]).T , 
			method = "splinef2d", bounds_error = False
	)

	#NaNs should have zero density
	z[np.where(np.isnan(z))] = 0.0
	z[z < 0] = 0
	return z


def dict_plot(dictionary, plot_func, figure_width=7.5, min_subplot_width=3.74,
			  aspect_ratio=.8, aliases={}, xlabel=None, ylabel=None, *args, **kwargs):
	'''
	A utility for generating a figure with a subplot for each entry in `dictionary`. 
	Parameters:
		`dictionary` (`dict`): The data to be plotted. The keys of the dictionary will be used
			as subplot titles.
		`plot_func` (callable): will be called as `plot_func(value, *args, **kwargs)` for each value in `
			dictionary`.
		`figure_width` (`float`: total width of the figure
		`min_subplot_width` (`float`): when laying out subplots, how narrow they are allowed to be.
		`aspect_ratio` (`float`): the ration of subplot height to width - not the same as matplotlib's
			definition
		`aliases` (`dict`):  optional alternative names to use as plot titles
		`xlabel`, `ylabel` (`str`): optional axis labels for the subplots
		Other args and kwargs passed to `plot_func`
	Returns:
		fig, axes: the matplotlib figure and subplots
	'''
	nplots = len(dictionary)
	plots_per_row = min(nplots, int(figure_width//min_subplot_width))
	nrows = int(np.ceil(nplots/plots_per_row))
	panel_width = figure_width/plots_per_row
	panel_height = panel_width * aspect_ratio
	figure_height = panel_height * nrows
	fig, axes = plt.subplots(nrows, plots_per_row, figsize=(figure_width, figure_height))
	if nrows > 1:
		axes = [a for ax in axes for a in ax]
	elif nplots == 1:
		axes = [axes]
	for key, ax in zip(dictionary.keys(), axes):
		plt.sca(ax)
		plot_func(dictionary[key], *args, **kwargs)
		if not xlabel is None:
			plt.xlabel(xlabel)
		if not ylabel is None:
			plt.ylabel(ylabel)
		if key in aliases:
			key = aliases[key]
		plt.title(key)
	plt.tight_layout()
	return fig, axes




def density_scatter(x, y, ax=None, sort=True, bins=50, trend_line=True, trend_line_args=dict(color='r'),
	lowess_args={}, diagonal=False, diagonal_kws=dict(color='black', lw=.3, linestyle='--'), 
	c="density", cbar_label=None,
	label_specific=[], label_outliers=0, outliers_from='trend', label_kws=dict(
					fontsize=8, color=(.3, 0, 0), 
					path_effects=[pe.withStroke(linewidth=-2, foreground=(1, 1, 1))]
					), 
	outlier_scatter_kws=dict(color=(.8, .2, .1), s=10, linewidth=.6, edgecolor=[0, 0, 0]), **kwargs ):
	"""
	Adapted from Guillaume's answer at
	 https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
	Scatter plot colored by 2d histogram, with optional trend_line, diagonal, and outlier labeling
	Parameters:
		`x`, `y`: `pandas.Series` with overlapping indices or iterables of the same length. Values to plot on each axis.
		`ax` (`matplotlib.Axis`): if provided, draw plot to this
		`sort` (`bool`): if `True` (default), the densest points are plotted last. 
		`bins` (`int`): How many bins to use in np.histogram2d for estimating density. Default 50.
		`trend_line` (`bool`): Whether to draw a lowess trend_line line
		`lowess_args` (`dict`): passed to `lowess_trend` for the trend_line line
		`trend_line_args` (`dict`): passed to `pyplot.plot` for the trend_line line
		`diagonal` (`bool`): If true, draw a line on the diagonal
		`diagonal_kws` (`dict`): Passed to `pyplot.plot`. By default, colors diagonal line red
		`label_outliers` (`int`): if > 0, the number of outliers to label with their index. 
				If `trend_line`, the outliers will be identified by deviation from the trend.
		'outliers_from':
			'trend': outliers identified by distance from trend line
			'diagonal': outliers identified by difference between `x` and `y`
			'density': outliers identified by minimum distance to plot region of high density
			'xy_zscore': outliers identified by euclidian distance from zero in z-score space
		`label_kws` (`dict`): passed to `pyplot.text` for the labels
		'outlier_scatter_kws': passed to `pyplot.scatter` to plot over outliers
		**kwargs: additional arguments passed to `pyplot.scatter`.
	"""
	if ax is None :
		fig = plt.gcf()
		ax = plt.gca()
	else:
		fig = ax.figure
	index = None
	if isinstance(x, pd.Series) and isinstance(y, pd.Series):
		x, y = x.align(y, join="inner")
		index = x.index
	if len(x) != len(y):
		raise ValueError("If not pd.Series, x and y must be the same length")
	mask = pd.notnull(x) & pd.notnull(y)
	x = np.array(x[mask]).astype(float)
	y = np.array(y[mask]).astype(float)
	if not index is None:
		index = index[mask]

	if c is "density" or outliers_from == "density":
		z = get_density(x, y, bins)
		z = np.sqrt(z)

	if c is "density":
		c = z
		if cbar_label is None:
			cbar_label = "Density (sqrt)"

	# Sort the points by c, so that the strongest points are plotted last
	if sort :
		idx = c.argsort()
		x, y, c = x[idx], y[idx], c[idx]
		if not index is None:
			index = index[idx]
		if c is "density" or outliers_from == "density":
			z = z[idx]

	im = ax.scatter( x, y, c=c, **kwargs )

	norm = Normalize(vmin = np.min(c), vmax = np.max(c))
	colormap = cm.ScalarMappable(norm = norm)
	colormap.set_array([])
	colormap.set_cmap(im.get_cmap())
	cbar = fig.colorbar(colormap, ax=ax)
	cbar.ax.set_ylabel(cbar_label)

	smoothed=None
	if trend_line:
		smoothed = lowess_trend(x, y, **lowess_args)
		xsort = np.argsort(x)
		ax.plot(x[xsort], smoothed[xsort], **trend_line_args)

	outliers = None
	if label_outliers:
		if outliers_from == 'trend':
			outliers = identify_outliers_by_trend(x, y, label_outliers, smoothed)
		elif outliers_from =='density':
			outliers = identify_outliers_by_density(x, y, z, label_outliers)
		elif outliers_from == 'diagonal':
			outliers = identify_outliers_by_diagonal(x, y, label_outliers)
		elif outliers_from == 'xy_zscore':
			outliers = identify_outliers_by_zscore(x, y, label_outliers)
		elif outliers_from == 'x':
			outliers = identify_outliers_1d(x, label_outliers)
		elif outliers_from == 'y':
			outliers = identify_outliers_1d(y, label_outliers)
		else:
			raise ValueError("`outliers_from` must be one of 'trend', 'density', 'diagonal', 'xy_zscore', 'x', or 'y'")

	if len(label_specific) and not index is None:
		label_specific = [index.get_loc(v) for v in label_specific]

	if not outliers is None:
		label_specific = sorted(set(label_specific) | set(outliers))

	if not index is None:
		labels = index[label_specific]
	else:
		labels = label_specific
	if len(label_specific):
		texts = [plt.text(s=labels[i], x=x[val], y=y[val], **label_kws)
				for i, val in enumerate(label_specific)]
		label_x = np.array([x[label] for label in label_specific])
		label_y = np.array([y[label] for label in label_specific])
		plt.scatter(label_x, label_y, **outlier_scatter_kws)
		if adjust_text_present and len(texts) > 0:
			adjust_text(texts, lim=500,
				arrowprops=dict(arrowstyle="-", color=[.7, .5, .5]),
				#x=outlier_x, y=outlier_y
				)
		elif len(texts) > 0:
			warn("adjustText not found. Install to have labels moved off points.")

	if not diagonal:
		return ax

	minmin = min(ax.get_xlim()[0], ax.get_ylim()[0])
	maxmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
	ax.plot([minmin, maxmax], [minmin, maxmax], **diagonal_kws)

	return ax


def binplot(x, y, binned_axis='x', nbins=10, endpoints=None, right=False, ax=None, colors=None, cbar_label='Number Samples', **kwargs):
	'''
	creates a plot with values binned into boxes along one axis. 
	Params:
		x: iterable of numbers indicating position on x axis
		y: iterable of numbers indicating position on y axis. 
		binned_axis (str): 'x' or 'y', the axis to bin ('x' default)
		nbins (int): number of discrete bins that will be created
		endpoints (None or tuple of two numbers): The right/top edge of the first bin and the left/bottom edge of the last bin. If provided,
				the first and last bins will include points in [-infinity, endpoints[0]] and [endpoints[1], +infinity] respectively. Other bins
				will be evenly spaced between them. If endpoints is None (default), bins will be evenly spaced between the minimum and maximum
				data points.
		right (bool): whether points falling on an edge are included in the left or right bin.
		axis (None or pyplot.Axis): axis to draw plot on (default or None draws to current axis)
		colors (None or str or iterable of RGBA values): color palette used to color the bins
	Additional keyword arguments are passed to pyplot.boxplot.
	'''
	if isinstance(x, pd.Series) and isinstance(y, pd.Series):
		x, y = x.align(y, join="inner")
		index = x.index
	mask = pd.notnull(x) & pd.notnull(y)
	x = np.array(x)[mask]
	y = np.array(y)[mask]
	if colors is None:
		colors = "viridis"
		
	if binned_axis == 'x':
		unbinned = 'y'
		vert = True
	elif binned_axis == 'y':
		unbinned = 'x'
		vert = False
	else:
		raise ValueError("binned_axis must be 'x' or 'y'")
		
	if isinstance(x, pd.Series) and isinstance(y, pd.Series):
		assert len(set(x.index) & set(y.index)) > 2, "x and y lack common indices"
	else:
		assert len(x) == len(y), "if x and y are not Series, they must be the same length"
		
	df = pd.DataFrame({'x': x, 'y': y})
	
	if endpoints is None:
		bins = np.linspace(df[binned_axis].min()-1e-12, df[binned_axis].max()+1e-12, nbins+1)
		space = bins[2] - bins[1]
		medians = .5*(bins[1:] + bins[:-1])
	else:
		bins = [-np.inf] + list(np.linspace(endpoints[0], endpoints[1], nbins-1)) + [np.inf]
		space = bins[2] - bins[1]
		medians = np.array(
			[bins[1] - .5*space] + list(.5*np.array(bins[2:-1]) + .5*np.array(bins[1:-2])) + [bins[-2] + .5*space]
		)
		
	digits = np.digitize(df[binned_axis], bins, right=right).astype(int)

	if any(digits > nbins):
		print(df[binned_axis][digits > nbins])
		assert False
	df[binned_axis] = medians[digits-1]
	
	if ax is None:
		ax=plt.gca()
	else:
		plt.sca(ax)
	vals = sorted([val for val in sorted(medians) if (df[binned_axis] == val).sum() > 0])
	
	boxes = plt.boxplot(x=[df[df[binned_axis] == val][unbinned]
				   for val in vals
				  ], 
				positions=vals, widths=[.9*space]*len(vals), patch_artist=True, vert=vert,

				**kwargs)

	counts = df[binned_axis].value_counts().reindex(index=medians).fillna(0)
	normer = LogNorm(vmin=0)
	normer.autoscale(counts.values)
	cvals = normer(counts.values)
	if isinstance(colors, str):
		cmap = colormaps[colors]
		colors = [cmap(v) for v in cvals]
	for box, color in zip(boxes['boxes'], colors):
		box.set_facecolor(color)

		
	if binned_axis == 'x':
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		if not endpoints is None:
			plt.xticks(bins[1:-1])
		else:
			plt.xticks(bins)
	else:
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		if not endpoints is None:
			plt.yticks(bins[1:-1])
		else:
			plt.yticks(bins)
	if binned_axis == 'x':
		plt.xlim(bins[1] - 1.2*space, bins[-2] + 1.2 * space)
	else:
		plt.ylim(bins[1] - 1.2*space, bins[-2] + 1.2 * space)

	try:
		mappable=ScalarMappable(norm=normer, cmap=cmap)
		mappable.set_array(colors)
		plt.gcf().colorbar(mappable, ax=plt.gca(), label=cbar_label)
	except:
		pass

	return ax