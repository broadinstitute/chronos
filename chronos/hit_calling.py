import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection, multipletests
import statsmodels.api as sm
from .model import Chronos, check_inputs, calculate_fold_change, normalize_readcounts, StdoutRedirector
from .reports import sum_collapse_dataframes
from .evaluations import fast_cor
from warnings import warn
from itertools import permutations
from scipy.stats import gaussian_kde, norm, lognorm, combine_pvalues, uniform, ks_1samp, linregress, genpareto
from scipy.stats import beta as beta_dist
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sympy.utilities.iterables import multiset_permutations
from shutil import rmtree
from tempfile import mkdtemp
import os
try:
	from tqdm import tqdm
except ImportError:
	def tqdm(x, *args, **kwargs):
		return x


def get_optimizer_noise_cutoff(log_likelihoods, fraction=.2):
	'''
	Return the least conservative value u0 such that
		total log_likelihoods < -u0 / (total log_likelihoods > u0 + total log_likelihoods < -u0) < fraction
	Negative changes in log-likelihood with increased parameterization can only occur due to failed
	optimization. Assuming the noise due to random final optimizer states is symmetric around zero,
	we treat the tail < 0 as the shape of the optimizer noise contribution to the distribution
	of likelihoods. Then we look for a value u0 large enough that fraction of the data with 0 true
	increase in likelihood explained by this distribution would <= `fraction` .
	Parameters:
		`log_likelihoods` (1D array): the change in log-likelihood of data with increased parameters
		`fraction` (float in [0, 1]): the acceptable amount of optimizer noise to include
	Returns:
		`u0` (`float` or `None`): the threshold above which the proportion of log-likelihoods attributable to
			0-true-increase-in-likelihood fraction operated on by optimizer noise is < `fraction`, if one exists
	'''
	log_likelihoods = np.sort(log_likelihoods)
	n_neg = (log_likelihoods < 0).sum()

	#find the number of negative log likelihoods 
	#more extreme than the ith-least-positive log likelihood:
	# the ith element is sum(null < -null[-i]) = sum(-null > null[-i])
	num_negatives_included = np.searchsorted(
	log_likelihoods[log_likelihoods < 0], 
		-log_likelihoods[::-1]
	)

	num_positives_included = np.arange(1, len(log_likelihoods)+1)
	ratio = num_negatives_included / (num_negatives_included + num_positives_included)
	#in this convention, we're counting backwards from the most positive value to find the first 
	#index where the ratio > fraction
	index = np.arange(len(log_likelihoods)).astype(int)[::-1]
	try:
		unmirrored_ind = np.arange(len(log_likelihoods))[ratio > fraction][0] - 1
		u0_ind = index[ratio > fraction][0] + 1
	except IndexError as e:
		if n_neg/len(log_likelihoods) < fraction and ratio[0] < fraction:
			return 0
		else:
			return None

	u0 = log_likelihoods[u0_ind]
	confirmed_ratio = np.sum(log_likelihoods < -u0) / (np.sum(log_likelihoods < -u0) + np.sum(log_likelihoods > u0))
	if  confirmed_ratio > 1.001*fraction:
		print(u0, confirmed_ratio,
			num_negatives_included[-u0_ind], np.sum(log_likelihoods < -u0),
			num_positives_included[-u0_ind], np.sum(log_likelihoods > u0),
			ratio[-u0_ind])
		raise RuntimeError("bug encountered that caused invalid u0 value")
	if u0 < 0:
		u0 = 0
	return u0


def genpareto_params(log_likelihoods, u0, clamp_xi=True):
	'''
	Get the generalized Pareto parameters that best fit the `log_likelihoods` tail starting at `u0`
	using maximum likelihood
	Parameters:
		`log_likelihoods` (1D array): the change in log-likelihood of data with increased parameters
		`u0` (`float` > 0): the point at which to use the generalized pareto
		`clamp_xi` (`bool`): if xi < 0 and this is true, return xi=0
	Returns:
		`p_u0` (`float` in [0, 1]): the baseline probability of data being in the tail modeled
			by the pareto
		`xi` (`float` > 0): the generalized Pareto shape parameter
		`sigma` (`float` > 0): the Pareto scale
	'''
	p_u0 = ((log_likelihoods > u0).sum() + 1) / (len(log_likelihoods)+1)
	y = log_likelihoods[log_likelihoods > u0]
	y = y - u0
	xi, _, sigma = genpareto.fit(y, floc=0)
	if clamp_xi:
		if xi < 0:
			warn(f"found negative generalized Pareto shape parameter xi={xi}, clipping to 0")
			xi = 0
	return p_u0, xi, sigma


def genpareto_certify(log_likelihoods, p_u0, xi, sigma, u0, alpha, delta=2.0, conf=0.95):
	'''
	Certify that the true frequency with which null data receive an estimated
	p-value below `alpha` is within a factor `delta` of `alpha`,
	with the rate of false certification controlled at `.5*(1-conf)`,
	assuming i.i.d-distributed `log_likelihoods` which are not used to learn 
	the generalized Pareto parameters.
	
	Parameters:
		`log_likelihoods` (1D array): held out data to test
		`p_u0` (`float` in (0, 1)): probability of log likelihoods exceeding threshold `u0` under null.
		`xi` (`float` > 0): generalized Pareto shape parameter, learned from data other than `log_likelihoods`.
		`sigma` (`float` > 0): generalized Pareto scale.
		`u0` (`float` > 0): value after which the tail of log_likelihoods are estimated by the generalized Pareto
		`alpha` (`float` in (0, p_u0)): the p-value at which to certify
		`delta` (`float` > 1): the allowed tolerance 
		`conf` (`float` in (0, 1)): the confidence interval to certify by
	Returns: `certification`, `min_delta`
		certification (`str`): 
			'certified' if the entire confidence interval is within the bounds of (alpha / delta, alpha * delta)
			'rejected' if the entire confidence interval falls outside the bounds of (alpha / delta, alpha * delta)
			'indeterminate' if the confidence interval overlaps one bound
			'underpowered' if the confidence interval overlaps both bounds
		`min_certifiable_delta` (`float` or `None`): the smallest error in p-value that can be certified with `conf` confidence
	'''
	assert 0 < p_u0 < 1, "p_u0 must be in (0, 1)"
	assert 0 <= xi , "xi must be positive"
	assert sigma > 0, "sigma must be positive"
	assert 0 < u0, "u0 must be positive"
	assert np.min(log_likelihoods) < u0 < np.max(log_likelihoods), \
		"log_likelihoods must include the full distribution with values above and below u0"
	assert 0 < alpha < p_u0, "alpha must be below the threshold's tail fraction and positive"
	assert 1 < delta, "delta must be > 1"
	assert 0 < conf < 1, "conf must be in (0, 1)"
	
	M   = len(log_likelihoods)
	exc = log_likelihoods[log_likelihoods > u0]
	P   = p_u0 * genpareto.sf(exc - u0, xi, scale=sigma)
	k   = int((P < alpha).sum())
	lo  = beta_dist.ppf((1-conf)/2, k, M-k+1) if k else 0.0
	hi  = beta_dist.ppf(1-(1-conf)/2, k+1, M-k)
	min_certifiable_delta = max(hi/alpha, alpha/(lo+1e-10))
	low_pass = lo/alpha > 1/delta
	high_pass = hi/alpha < delta
	
	if low_pass and high_pass:
		certification = "certified"
	elif hi/alpha < 1/delta or lo/alpha > delta:
		certification = "rejected"
	elif lo/alpha < 1/delta and hi/alpha > delta:
		certification = 'underpowered'
	else:
		certification = "indeterminate"

	if certification == "rejected":
		min_certifiable_delta = None
	return certification, min_certifiable_delta


def genpareto_p(log_likelihoods, p_u0, xi, sigma, u0):
	'''
	Estimate the p-values for the log_likelihoods > u0, using
	the generalized Pareto distribution with xi and sigma:
	p(delta_LL) = p_u0 * Pareto(delta_LL - u0 | xi, sigma) if delta_LL > u0
				= NaN, otherwise
	Parameters:
		`log_likelihoods` (1D array): change in log likelihood
		`p_u0` (`float` in (0, 1)): probability of log likelihoods exceeding threshold `u0` under null.
		`xi` (`float` >= 0): generalized Pareto shape parameter, learned from data other than `log_likelihoods`.
		`sigma` (`float` > 0): generalized Pareto scale.
		`u0` (`float` > 0): value after which the tail of log_likelihoods are estimated by the generalized Pareto
	Returns:
		`p` (1D array with shape matching `log_likelihoods`): gives the p-values for log_likelihoods > u0, np.nan
			for the rest.
	'''
	assert 0 < p_u0 < 1, "p_u0 must be in (0, 1)"
	assert 0 <= xi , "xi must be >= 0"
	assert sigma > 0, "sigma must be positive"
	assert 0 < u0, "u0 must be positive"
	
	p = np.full(len(log_likelihoods), np.nan, dtype=float)
	mask = log_likelihoods > u0
	estimates = p_u0 * genpareto.sf(log_likelihoods[mask] - u0, c=xi, scale=sigma)
	if (estimates < 0).any():
		raise ValueError(f"{(estimates < 0).sum()} negative generalized Pareto p-values found")
	p[mask] = estimates
	if (p < 0).any():
		raise ValueError(f"{(p < 0).sum()} negative generalized Pareto p-values found")
	return p


class IncompatibleBoundariesError(Exception):
	pass

def get_u0(log_likelihoods, noise_fraction=.2, min_samples_retain=300,
		  violation="warn"):
	'''
	Find a boundary for fitting the generalized Pareto distribution which is:
		- large enough that no more than `noise_fraction` of the data above the tail
			could be explained by data with 0 true increase in log-likehood amplified by 
			optimizer noise
		- is at least above the `lowest_quantile` of the data
		- but has at least `min_samples_retain`
	Parameters:
		`log_likelihoods` (1D array): the change in log-likelihood of data with increased parameters
		`noise_fraction` (float in [0, 1]): the acceptable amount of optimizer noise to tolerate in the tail
		`min_samples_retain` (`int` > 0): the minimum number of samples to aim for in the tail
		`violation` ("warn", "error", "ignore"): how to respond to cases where no u0 can be found 
			satisfying all conditions
		returns:
			`u0` (`float` or `None`): the chosen boundary
	'''
	optimizer_bound = get_optimizer_noise_cutoff(log_likelihoods, noise_fraction)
	if optimizer_bound is None:
		optimizer_bound = np.inf
	u0_min = optimizer_bound
	u0_max = np.sort(log_likelihoods)[-min_samples_retain-1]

	if u0_min > u0_max:
		message = (
			"not enough data to meet all conditions on the boundary u0 for fitting Pareto, "
			"or too much optimization noise. You can allow larger noise_fraction "
			"increase the number of training steps, or decrease regularization. "
			f"Total samples (points): {len(log_likelihoods)}, "
			f"Minimum bound for controlling optimizer noise below "
			f"{noise_fraction}: {optimizer_bound:.3f}. "
			f"Maximum value to include at least {min_samples_retain} points for fitting: "
			f"{u0_max:.3f}"
		)

		if violation == "warn":
			warn(message)
		elif violation == "error":
			raise IncompatibleBoundariesError(message)
		elif violation == "ignore":
			pass
		else:
			raise ValueError("`violation` must be one of 'warn', 'error', 'ignore'")
	return u0_max


def genpareto_heldout_certification(permuted_lls, test_frac=.5, noise_fraction=.2, delta=2.0, conf=.95, 
									alpha_rank=50, min_samples_retain=300,
									seed=None):
	'''
	Check whether a generalized Pareto model is an acceptable method for estimating tail p-values.
	Splits the data into 1-test_frac, test_frac randomly, learns the parameters from the train fraction,
	and then certifies the model on test fracton. 
	
	Parameters:
		`permuted_lls` (1D array): null log-likelihood increases
		`test_frac` (`float` in (0, 1)): fraction data to reserve for certifying.
		`noise_fraction` (`float` in (0, 1)): allowable fraction of likelihood changes attributable to optimizer noise
			in the tail that will be fit by generalized Pareto
		`delta` (`float` > 1): the allowed tolerance 
		`conf` (`float` in (0, 1)): the confidence interval to certify by
		`alpha_rank` (`int` > 0): the number of points in the train set to use for certification:
			the estimated p-value of the permuted_ll in the test set at this rank (counting back from last)
			is what will be certified. Increasing `alpha_rank` gives the test more power and reduces the chance 
			of getting indeterminate results. The price is less protection for the extreme tail values where 
			the accuracy of the estimate is most important.
		`min_samples_retain` (`int` > 0): the minimum number of samples to aim for in the tail
		`seed` (`int`): controls the random train/test split with numpy
	Returns: `certification`, `alpha`, `min_certifiable_delta`
		certification (`str`):
			the certification result from genpareto_certify, and the alpha that was certified. Automatically
				rejects if it finds the optimal shape parameter `xi` <= 0.
		`alpha` (`float` in [0, 1] or `None`): the estimate p-value which is certified
		`min_delta` (`float`): the smallest relative error in p-value that can be certified with confidence `conf`
	'''
	rng = np.random.default_rng(seed=seed)
	cutoff = int(np.round((1-test_frac) * len(permuted_lls)))
	perm = rng.permutation(len(permuted_lls))
	train, test = permuted_lls[perm[:cutoff]], permuted_lls[perm[cutoff:]]
	if len(train) < 1000:
		warn(
			f"only {len(train)} samples in the training set for certifying "
			"the generalized Pareto distribution."
		)
	u0 = get_u0(
			train, noise_fraction, min_samples_retain, violation="error"
	)
	if u0 is None:
		return "rejected", None, None
	p_u0, xi, sigma = genpareto_params(train, u0)
	if xi < 0:
		return "rejected", np.nan, np.nan
	alpha_rank = min(alpha_rank, np.sum(test > u0).astype(int))
	alpha = alpha_rank / len(test)
	certification, min_delta = genpareto_certify(test, p_u0, xi, sigma, u0, alpha, delta, conf)
	return certification, alpha, min_delta


def loglikelihood_p(observed_lls, permuted_lls, noise_fraction=.2, test_frac=.5, 
					delta=2, conf=.95, alpha_rank=50, accept_underpowered=True, accept_indeterminate=True,
					min_samples_for_genpareto=300, 
					seed=None
	):
	'''
	Computes estimated p-values for the observed_lls using the permuted_lls as the null.
	Attempts to find a threshold above which permuted_lls are mostly not explained by optimizer noise ( 
		tolerating `noise_fraction` of the data to still be so explained). For values below the threshold,
		p is estimated empirically using the fraction of permuted_lls > the sample being tested. Above this
		threshold, p is estimated using the generalized Pareto distribution. Various issues can cause the 
		fit to the generalized Pareto to be discarded, usually boiling down to very noise or insufficient 
		permuted_lls. In this case, all the p-values are estimated empirically and a warning is raised.
	Parameters:
		`observed_lls` (1D array): true log-likelihood increases
		`permuted_lls` (1D array): null log-likelihood increases
		`noise_fraction` (`float` in (0, 1)): allowable fraction of likelihood changes attributable to optimizer noise
			in the tail that will be fit by generalized Pareto
		`test_frac` (`float` in (0, 1)): fraction data to reserve for certifying.
		`delta` (`float` > 1): the acceptable ratio error in p-values using the generalized Pareto distribution, see below. 
		`conf` (`float` in (0, 1)): The confidence interval for certifying the generalized Pareto distribution; the opposite
			of the acceptable Type I error rate in certification.
			If:
				the true proportion of *null* values with Pareto_estimated p < alpha
				differs from alpha 
				by more than delta (i.e. alpha_true > alpha * delta, alpha_true < alpha / delta)
				(in the limit of infinite data)
				(for some chosen certification point alpha)
			Then:
				this function will falsely certify the Pareto distribution for use in *no more than* `.5*(1-conf)` cases.
		`alpha_rank` (`int` > 0): the number of points in the train set to use for certification:
			the estimated p-value of the permuted_ll in the test set at this rank (counting back from last)
			is what will be certified. Increasing `alpha_rank` gives the test more power and reduces the chance 
			of getting indeterminate results. The price is less protection for the extreme tail values where 
			the accuracy of the estimate is most important.
		`accept_indeterminate` (`bool`): 
			In the case that the acceptable range of errors in p partially overlaps the confidence interval,
			use Pareto anyway (recommended)
		`accept_underpowered` (`bool`): 
			In the case that the confidence interval is too wide to certify at chosen p, use Pareto anyway (recommended)
		`seed` (`int`): controls the random train/test split with numpy, used for certification
	Returns:
		`p`, `certification`, `alpha`, `min_delta`, `u0`, `disagreement`: 
			`p` (1D array matching `observed_lls`): the combined empirical/Pareto p-values
			`certification` (`str`): the certification result from genpareto_certify
			`alpha` (`float` in [0, 1] or `None`): the estimated p-value that was certified within `delta` tolerance.
			`min_delta` (`float`): the smallest relative error in p-value that can be certified with confidence `conf`
				for the Pareto distribution. Pessimistic since it is based on fewer null samples than you actually have.
			`u0` (`float` > 0): the threshold on observed_lls. Likelihoods above this threshold were assigned p-values
				by the Pareto model, unless `certification` is "rejected".
			`disagreement` (`float`): the RMSE of log(Pareto estimated p) - log(empirical p)
			`bias` (`float`): the mean ratio of Pareto estimated to empirical p in up to the last 50 values: 
				exp(mean(log(Pareto estimated p) - log(empirical p)))
	'''
	p_emp = empirical_pvalue(observed_lls, permuted_lls, direction=1)
	genpareto_fit_properties = dict(
		certification="rejected", 
		alpha=np.nan, min_delta=np.nan, u0=np.nan, 
		disagreement=np.nan, bias=np.nan,
		xi=np.nan, sigma=np.nan, p_u0=np.nan
	)
	if (p_emp < 0).any():
		raise ValueError("negative empirical p-values")
	if len(permuted_lls) < min_samples_for_genpareto:
		warn(
			"Too few total values in the permuted log likelihood changes to estimate "
			"a null distribution. "
			"Only empirical p-values will be returned."
		)
		return p_emp, genpareto_fit_properties

	try:
		u0 = get_u0(permuted_lls, noise_fraction, min_samples_for_genpareto, violation="error")
	except IncompatibleBoundariesError as e:
		warn("not enough permuted data to find a good tail for generalized Pareto. "
			 "Only empirical p-values will be returned. Details for the tail cutoff:\n"
			 f"{e}"
			)
		return p_emp, genpareto_fit_properties
	genpareto_fit_properties["u0"] = u0
	n_samples = (permuted_lls > u0).sum()
	if n_samples <  min_samples_for_genpareto:
		warn(
			"Too much of the permuted likelihoods is explained by pure optimization noise "
			"(indicated by the large number of negative changes in likelihood extending into "
			"the tail). "
			f"After excluding the bulk explained by optimization noise, {n_samples} "
			"points remain. "
			"Only empirical p-values will be returned."
		)
		return p_emp, genpareto_fit_properties

	try:
		(
			genpareto_fit_properties["certification"], 
			genpareto_fit_properties["alpha"], 
			genpareto_fit_properties["delta"]
		) = genpareto_heldout_certification(
				permuted_lls, test_frac, noise_fraction, delta, conf, alpha_rank, 
				min_samples_for_genpareto,
				seed
		)
	except IncompatibleBoundariesError:
		warn(
			"Could not satisfy all the requested boundaries on the tail threshold "
			"to switch to generalized Pareto while certifying."
		)
		genpareto_fit_properties["certification"] = "indeterminate"
		genpareto_fit_properties["alpha"] = np.nan
		genpareto_fit_properties["min_delta"] = np.nan

	if genpareto_fit_properties["certification"] == "rejected":
		warn(
			f'the generalized Pareto p estimates were not certified ("{certification}"). '
			f'Only empirical p-values will be returned. '
			f'You can reduce `conf` from {conf} to tolerate a higher rate of false certification '
			f'or increase `delta` from {delta} to at least {min_delta} '
			f'to tolerate greater error in certified p-values.'
		)
		return p_emp, genpareto_fit_properties

	if genpareto_fit_properties["certification"] == "indeterminate" and not accept_indeterminate:
		warn(
			f'the generalized Pareto p estimates could not be certified, '
			f'possibly due to insufficient data. '
			f'Only empirical p-values will be returned. '
			f'Pass `accept_indeterminate=True (recommended) '
			f'to use the generalized Pareto model anyway '
			f'You can reduce `conf` from {conf} to tolerate a higher rate of false certification '
			f'or increase `delta` from {delta} to at least {min_delta} '
			f'to tolerate greater error in certified p-values.'
		)
		return p_emp, genpareto_fit_properties
	if genpareto_fit_properties["certification"] == "underpowered" and not accept_underpowered:
		warn(
			f'the generalized Pareto p estimates could not be certified due to insufficient data '
			f'Only empirical p-values will be returned. '
			f'Pass `accept_underpowered=True (recommended) ' 
			f'to use the generalized Pareto model anyway '
			f'You can reduce `conf` from {conf} to tolerate a higher rate of false certification '
			f'or increase `delta` from {delta} to at least {min_delta} '
			f'to tolerate greater error in certified p-values.'
		)
		return p_emp, genpareto_fit_properties

	(
			genpareto_fit_properties["p_u0"], 
			genpareto_fit_properties["xi"], 
			genpareto_fit_properties["sigma"]
	) = genpareto_params(permuted_lls, u0)
	if genpareto_fit_properties["xi"] < 0:
		warn(
			f"inferred an invalid generalized pareto shape parameter xi={genpareto_fit_properties["xi"]}. "
			f"Only empirical p-values will be returned." 
		)
		genpareto_fit_properties["certification"] = "rejected"
		return p_emp, genpareto_fit_properties

	p_est = genpareto_p(observed_lls, **{key: genpareto_fit_properties[key] for key in ['p_u0', 'xi', 'sigma', 'u0']})
	p_composite = np.where(pd.notnull(p_est), p_est, p_emp)
	mask = (observed_lls > u0) & (observed_lls < permuted_lls.max())
	disagreement = np.sqrt(np.mean(np.square(np.log(p_est[mask]) - np.log(p_emp[mask]))))
	mask = mask & (observed_lls > np.sort(observed_lls)[-50])
	bias = np.exp(np.mean(np.log(p_est[mask]) - np.log(p_emp[mask])))
	genpareto_fit_properties["disagreement"] = disagreement
	genpareto_fit_properties["bias"] = bias
	return p_composite, genpareto_fit_properties



def cell_line_log_likelihood(model, distinguished_condition_map):
	'''get the log likelihood of the data from the Chronos model summed to the cell line/gene level and combined
	over libraries. Note the negative sign, since `Chronos.cost_presum` is the negative log likelihood'''
	cost_presum = model.cost_presum
	cost_presum = [
		v\
			.groupby(distinguished_condition_map[key].set_index("sequence_ID")['true_cell_line_name']).sum()\
			.T.groupby(model.guide_gene_map[key].set_index("sgrna")["gene"]).sum().T

		for key, v in cost_presum.items()
	]
	return -sum_collapse_dataframes(cost_presum)


def unique_permutations(array):
	out = list(np.array([v for v in multiset_permutations(array)]))
	#exclude the trivial permutation
	return [v for v in out if not np.all(v == array)]


def unique_permutations_no_reversed(array):
	unique = np.unique(array)
	assert len(unique) == 2, "Array must have exactly two unique values"
	mapper = {val: bool(i) for i, val in enumerate(unique)}
	bool_array = np.array([mapper[val] for val in array])
	perms = unique_permutations(bool_array)
	out = []
	mirror = set([
		tuple(bool_array),
		tuple(~bool_array)
		])
	for perm in perms:
		reverse = tuple(~perm)
		if not reverse in mirror:
			mirror.add(tuple(perm))
			out.append(unique[perm.astype(int)])
	return out



def reverse_gene_effect(gene_effect, distinguished_condition_map, condition_pair):
	'''reverse the condition labels for the gene effect matrix index'''
	out = gene_effect.copy()

	if isinstance(distinguished_condition_map, dict):
		lines = sorted(set.union(*[set(v.true_cell_line_name) 
			for v in distinguished_condition_map.values()]))
	else:
		lines = list(distinguished_condition_map.true_cell_line_name.unique())
	lines.remove("pDNA")

	for line in lines:

		out.loc[
			'%s__in__%s' % (line, condition_pair[0])
		] = gene_effect.loc['%s__in__%s' % (line, condition_pair[1])]

		out.loc[
			'%s__in__%s' % (line, condition_pair[1])
		] = gene_effect.loc['%s__in__%s' % (line, condition_pair[0])]

	return out


def change_in_likelihood(model, distinguished_condition_map, condition_pair):
	'''
	Get the change in likelihood for a `model` when the gene effect of each cell line
	in each condition is reversed
	'''
	likelihood_baseline = cell_line_log_likelihood(model, distinguished_condition_map)
	ge = model.gene_effect
	model.gene_effect = reverse_gene_effect(ge, distinguished_condition_map, condition_pair)
	likelihood_reversed = cell_line_log_likelihood(model, distinguished_condition_map)
	model.gene_effect = ge
	return likelihood_baseline - likelihood_reversed


def change_in_gene_effect(model, distinguished_condition_map, condition_pair):
	'''get the differences in gene effect for the same lines in the first vs second condition'''
	ge = model.gene_effect
	# get unique lines allowing for someone deciding to put "__in__" in their base cell line name
	all_lines = set(['__in__'.join(s.split('__in__')[:-1]) for s in ge.index])
	out = {
		line: ge.loc["%s__in__%s" % (line, condition_pair[1])]
			- ge.loc["%s__in__%s" % (line, condition_pair[0])]
		for line in all_lines
	}
	return pd.DataFrame(out).T



def empirical_pvalue(observed, null, direction=-1):
	'''
	Returns an array or `pandas.Series` of non-parametric pvalues given a null. 
	For a given observation in `observed`, `p = (n_null_more_extreme + 1) / (n_null + 1)`,
	where n_null_more_extreme is the number of null observations equal or more extreme
	that the observation and n_null is `len(null)`.
	Parameters:
		`observed` (1D array-like): observed values for the samples
		`null` (1D array-like): values sampled from the null hypothesis
		`direction` (-1 or 1), which tail is being tested. -1 tests the hyothesis that 
		the observed values are less positive than would be expected by chance
	Returns:
		numpy.ndarray or pandas.Series of pvalues
	'''
	if direction not in [-1, 1]:
		raise ValueError("`direction` must be -1 or 1")

	observed_array = -direction*np.array(observed)
	null_array = -direction*np.array(null)
	sort_observed = np.argsort(observed_array)
	sort_null = np.argsort(null_array)
	observed_array = observed_array[sort_observed]
	null_array = null_array[sort_null]

	pvals = (1+np.searchsorted(null_array, observed_array, side="right"))/(1+len(null_array))
	#undo the sort to get pvals in the same order as observed
	pvals = pvals[np.argsort(sort_observed)]
	if isinstance(observed, pd.Series):
		pvals = pd.Series(pvals, index=observed.index)
	return pvals


def empirical_pvalue_lognorm_extension(observed, null, direction=-1):
	'''
	computed an empirical p-value, then a lognorm likelihood p. For values in `observed` more extreme than any value in 
	`null`, the lognorm likelihood p-value is substituted.
	Parameters:
		`observed` (1D array-like): observed values for the samples
		`null` (1D array-like): values sampled from the null hypothesis
		`direction` (-1 or 1), which tail is being tested. -1 tests the hyothesis that 
		the observed values are less positive than would be expected by chance
	Returns:
		numpy.ndarray or pandas.Series of pvalues
	'''

	p = empirical_pvalue(observed, null, direction)
	# use the lognormal model for the null to extend p-values beyond most extreme null value
	intercept, s = fit_weighted_lognorm(null)
	p2 = lognorm_likelihood_p(observed, intercept, s, direction)
	if direction == 1:
		p.mask(observed > null.max(), p2, inplace=True)
	else:
		p.mask(observed < null.min(), p2, inplace=True)
	return p

class PNotCorrectedError(Exception):
	pass


def adjust_p_values(p_values, negative_control_index):
	'''
	Attempts to adjust `p_values` so that the negative controls are uniformly distributed in the left tail. 
	This is done by performing OLS regression n the negative control p-values vs the expected uniform 
	quantiles in log space, then predicting the expected deviation from uniformity for all p-values
	and returning the residuals. 
	'''
	logged_p = np.log(p_values.clip(1e-16, 1)).sort_values()
	logged_neg = logged_p.reindex(negative_control_index).dropna().sort_values()
	if not len(logged_neg):
		raise IndexError("no non-null values for the negative control index in the p_values")
	grid = np.log(np.linspace(1/len(logged_neg), len(logged_neg)/(len(logged_neg)+1), len(logged_neg)))
	lr = linregress(grid, logged_neg)
	good_fit = lr.rvalue > .99 and lr.pvalue < .01
	if not good_fit:
		raise PNotCorrectedError(f"bad fit for regression in log space on negative control p values vs \
the uniform distibution. Fit result:\n{lr}")
	full_grid = np.log(np.linspace(1/len(logged_p), len(logged_p)/(len(logged_p)+1), len(logged_p)))
	p_adjust = np.exp(logged_p - ((lr.slope-1) * full_grid + lr.intercept)).clip(0, 1)
	return p_adjust


################################################################
# C O M P A R E    C O N D I T I O N S
################################################################


def get_difference_significance(observed, null, tail, method="FDR_TSBH"):
	'''
	Combines effect size, p-value, and FDR in one dataframe
	'''
	if tail == "both":
		pvals = 2*np.minimum(
				empirical_pvalue(observed, null, direction=-1), 
				empirical_pvalue(observed, null, direction=1)
		)
	elif tail == "right":
		pvals = empirical_pvalue(observed, null, direction=1)
	elif tail == "left":
		pvals = empirical_pvalue(observed, null, direction=-1)
	else:
		raise ValueError("`tail` must be one of 'left', 'right', 'both'")

	fdr = multipletests(pvals, .05, method=method)[1]
	return pd.DataFrame({"observed_statistic": observed, "pval": pvals, "FDR": fdr})


def filter_sequence_map_by_condition(condition_map, condition_pair=None):
	'''
	Given a pair of conditions, removes cell lines from the `condition_map` without
	at least two replicates per condition and returns the new map. Replicates are filtered
	so that there is an even and equal number in each condition.
	'''
	out = condition_map.copy()
		
	if condition_pair is None:
		condition_pair = out.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")
	if set(condition_pair) - set(out.condition):
		raise ValueError("one or more entries in `condition_pair` %r not present in `condition_map.condition`" 
			% list(condition_pair))

	condition_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby("cell_line_name")\
					.condition\
					.nunique()
	if (condition_counts < 2).all():
		raise ValueError("No cell lines in `condition_map` had replicates in both conditions \
in `condition_pair`: %r/n/n%r" % (condition_pair, condition_map))
	if (condition_counts < 2).any():
		to_drop = condition_counts.loc[lambda x: x!=2].index
		warn("the following cell lines did not have replicates in both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]
		
	replicate_counts = out\
					.query("condition in %r" % list(condition_pair))\
					.query("cell_line_name != 'pDNA'")\
					.groupby(["cell_line_name", "condition"])\
					.replicate\
					.nunique()
	if (replicate_counts < 2).all():
		raise ValueError("No cell lines in `condition_map` had at least 2 replicates in both conditions \
in `condition_pair`: %r/n/n%r" % (condition_pair, condition_map))
	if (replicate_counts < 2).any():
		to_drop = set(replicate_counts.loc[lambda x: x<2].index.get_level_values(0))
		warn("the following cell lines did not have at least 2 replicates for both conditions and are being dropped:\n%r" % sorted(to_drop))
		out = out[~out.cell_line_name.isin(to_drop)]

	out = out[out.condition.isin(condition_pair) | (out.cell_line_name == 'pDNA')]

	#balance the number of replicates in each condition
	retain_sequences = []
	for line, group in out.groupby("cell_line_name"):
		if line == 'pDNA':
			retain_sequences.extend(list(group.sequence_ID))
			continue
		n_replicates = group.groupby("condition").replicate.nunique().min()
		if n_replicates % 2:
			n_replicates -= 1
		for condition, subgroup in group.groupby("condition"):
			retain_replicates = subgroup.replicate.unique()[:n_replicates]
			retain_sequences.extend(list(
				subgroup\
					.query("replicate in %r" % list(retain_replicates))\
					.sequence_ID
			))

	return out[out.sequence_ID.isin(retain_sequences)].copy()


def assign_condition_replicate_ID(condition_map):
	'''
	Create a condition replicate ID.
	'''
	condition_map["replicate_ID"] = condition_map.apply(
					lambda x: '%s__IN__%s_%s_%s' % (
						x["cell_line_name"], x["condition"], x["replicate"], x['pDNA_batch']
					),
					axis=1
				)


def create_condition_sequence_map(condition_map, condition_pair=None):
	'''
	Returns a new map filtered for `condition` in `condition_pair` with cell lines having fewer
	than 2 replicates in either condition removed, where `cell_line_name` has "__in__<condition>" appended.
	'''
	out = condition_map.copy()
	out['true_cell_line_name'] = out['cell_line_name'].copy()
	
	def cell_line_overwrite(x):
		if x.cell_line_name == 'pDNA':
			return 'pDNA'
		return '%s__in__%s' % (x.cell_line_name, x.condition)
	out['cell_line_name'] = out.apply(cell_line_overwrite, axis=1)
	
	return out

def _balance_permutation(perm):
	'''
	perm: an array with values which may be repeated
	if all unique values already have equal counts, does nothing. Otherwise, randomly selected instances of
		each unique value replaced with NaNs until balanced.
	'''
	counts = pd.Series(perm).value_counts().drop(np.nan, errors="ignore")
	counts_min = counts.min()
	for ind in counts.index:
		indices = np.argwhere(perm == ind).flatten()
		perm[np.random.choice(indices, size=counts[ind] - counts_min, replace=False)] = np.nan

def create_permuted_sequence_maps(condition_map, condition_pair=None, allow_reverse=False):
	'''
	Returns a list of condition maps, where each cell line within each map has a unique permutation
	of condition labels which have been appended to `cell_line_name`. The original condition and its 
	mirror image are never returned. If `allow_reverse` is False,
	mirroring permutations are discarded. E.g. if `condition` in `condition_map` is ['A', 'A', 'B', 'B'],
	only ['A', 'B', 'A', 'B'] and ['A', 'B', 'B', 'A'] are returned.
	Additionally, only permutations with an equal number of replicates from each condition are retained.
	'''
	seq_map = filter_sequence_map_by_condition(condition_map, condition_pair)

	if condition_pair is None:
		condition_pair = seq_map.query("cell_line_name != 'pDNA'").condition.unique()
	if len(condition_pair) != 2:
		raise ValueError("can only compare two conditions. If `condition_pair` is not passed, \
the 'condition' column of `condition_map` must have exactly two unique values for non-pDNA entries.")

	#drop days column for identifying possible permutations
	base = seq_map[["cell_line_name", "condition", "replicate_ID", "pDNA_batch"]].drop_duplicates()
	
	splits = base.groupby('cell_line_name')
	stack = {}
	for line, y, in splits:
		y['cell_line_name'] = line
		
		if line == 'pDNA':
			pdna = y.copy()
			pdna["true_condition"] = pdna["condition"]
			continue

		stack[line] = []
		if allow_reverse:
			perms = unique_permutations(y.condition)
		else:
			perms = unique_permutations_no_reversed(y.condition)

		for perm in perms:
			# randomly trim excess replicates of the more common condition
			# so total replicates balance - otherwise impossible to construct
			# two pseudo conditions with the same number of replicates from each 
			# real condition
			_balance_permutation(perm)

		for perm in perms:
			tentative = y[pd.notnull(perm)].copy()
			perm = perm[pd.notnull(perm)]
			tentative["true_condition"] = tentative["condition"]
			tentative["condition"] = perm
			tentative["cell_line_name"] = line

			#check that the pseudo conditions have equal numbers of replicates from each real condition
			condition_counts = tentative.groupby("condition").true_condition.value_counts()
			if \
					len(condition_counts) == 4 \
				 and condition_counts.min() == condition_counts.max():
				stack[line].append(tentative)

	min_unique_permutations = min(len(v) for v in stack.values())
	out = []
	for i in range(min_unique_permutations):
		#inner merge is to incl. row for each day (in case replicate has data for multiple days)
		#outer merge is to add back sequence ids for each row
		out.append(
			pd.merge(
				pd.concat([v[i] for v in stack.values()] + [pdna]), 
				seq_map.rename(columns={'condition':'true_condition'}),
				how="outer" 
			)
		) 

	return [create_condition_sequence_map(v, condition_pair) for v in out]


def check_condition_map(condition_map):
	expected_columns = ['sequence_ID', 'cell_line_name', 'pDNA_batch', 'days', 'replicate', 'condition']
	for key in condition_map.keys():
		missing = sorted(set(expected_columns) - set(condition_map[key].columns))
		if missing:
			raise ValueError(
				"`condition_map[%s]` missing expected columns %r" % (key, missing)
			)

def _remap_growth_rates(growth_rate, old_map, new_map):
	
	growth_rates = []
	
	for library, mapper in old_map.items():
		sub = growth_rate.query("library == @library").copy()
		mapper = mapper[["sequence_ID", "replicate_ID"]].drop_duplicates(subset=["replicate_ID"])
		sub = pd.merge(sub, mapper, on="replicate_ID", how="left")
		sub["replicate_ID"] = new_map\
			[library]\
			.set_index("sequence_ID")\
			["replicate_ID"]\
			.loc[sub.sequence_ID]\
			.values
		growth_rates.append(sub)
		
	return(pd.concat(growth_rates, ignore_index=True))


def check_for_excess_correlation(readcounts, condition_map, negative_control_sgrnas):
	warned = False

	for library in readcounts:

		library_readcounts = readcounts[library]
		library_condition_map = condition_map[library]
		library_negs = negative_control_sgrnas[library]
		normed = normalize_readcounts(
			library_readcounts, 
			negative_control_sgrnas=library_negs, 
			sequence_map=library_condition_map
		)
		lfc = np.log2(calculate_fold_change(
			normed, 
			sequence_map=library_condition_map, 
			rpm_normalize=False
		))

		cell_groups=library_condition_map.groupby("cell_line_name")
		mean_corrs = []

		for cell_line, cell_group in cell_groups:

			if cell_line == 'pDNA':
				continue
			corrs = fast_cor(
				lfc.T.loc[library_negs, cell_group.sequence_ID]
			)
			np.fill_diagonal(corrs.values, np.nan)

			day_groups = cell_group.groupby("days")

			for days, day_group in day_groups:
				group_conditions = day_group.condition.unique()
				for i, condition1 in enumerate(group_conditions):
					for condition2 in group_conditions[i:]:
						rows = day_group.query(f'condition == "{condition1}"').sequence_ID
						columns = day_group.query(f'condition == "{condition2}"').sequence_ID
						corr_subset = corrs.loc[rows, columns]
						mean_corrs.append(pd.Series({
							"cell_line_name": cell_line,
							"condition_pair": (condition1, condition2),
							"same_condition": condition1 == condition2,
							"days": days,
							"mean_corr": corr_subset.mean().mean()
						}))
		mean_corrs = pd.DataFrame(mean_corrs)
		diff_max = mean_corrs\
			.groupby(["cell_line_name", "days"])\
			.apply(lambda df: 
				   df[df.same_condition].mean_corr.max() - df[~df.same_condition].mean_corr.max(),
				   include_groups=False
			)
		if diff_max.max() > .1:
			warn("Library %s: Negative controls are more highly correlated between replicates of the same \
condition than between conditions for one or more cell lines/conditions/time points:\n\n%r\n\n \
Since we don't expect negative controls to have real viability effects, this implies that the biological \
replicates are not genuinely independent measurements, and p-values will be optimistic. \
This usually indicates that the replicates were not infected separately, and were instead \
infected before splitting. \
Chronos will try to adjust p-values to compensate for this effect, but the results are \
not guaranteed to be calibrated. A large, high quality set of negative control genes are \
essential for the adjustment. The correction should be robust if one or two of the negative \
control genes turn out to be genuinely differential between experiments, so prioritize a larger \
set over an extremely high confidence set of negative controls." 
						  % (library, mean_corrs.drop("same_condition", axis=1))
						 )
			warned = True

	return warned


def check_calibration(pvals, max_allowed_ks_statistic=.1):
	result = ks_1samp(pvals, uniform.cdf)
	if result.statistic > max_allowed_ks_statistic and result.pvalue < .05:
		return False
	else:
		return True

class ConditionComparison():
	'''
	An object that manages the various Chronos models needed to compare two conditions. 
	The strategy is to compare the chosen measure of difference between the two conditions
	per gene to a null distribution generated by permuting which replicates in each cell line
	are assigned to each condition, generating an empirical p-value. The key method that returns
	the statistics is `compare_conditions`.
	'''
	comparison_effect_dict = {
			"gene_effect": change_in_gene_effect,
			"likelihood": change_in_likelihood
	}
	# which tails to test the p-value in for each comparison effect
	comparison_effect_tail_dict = {
			"gene_effect": "both",
			"likelihood": "right"
	}

	def __init__(self, readcounts, condition_map, guide_gene_map,
		negative_control_genes=None, negative_control_sgrnas=None,
		keep_models=False, 
		intermediate_print_to='stdout', print_to=None, **kwargs):
		'''
		Initialize the comparator.
		Parameters:
			`readcounts` (`dict` of `pandas.DataFrame`): readcount matrices from the experiment.
					See `model.Chronos`
			`condition_map` (`dict` of `pandas.DataFrame`): Tables in the same format as `sequence_map`
					for `model.Chronos`, but now requires `replicate` (e.g. A, B), and 
					`condition`, which the comparator will compare results between. `condition` can be 
					any value that can be passed to `str`.
					Results will be reported separately per cell line.
					If you wish to compare two cell lines, give them the same value in `cell_line_name`,
					and different values for `condition`.
			`guide_gene_map` (`dict` of `pandas.DataFrame`): map from sgRNAs to genes. See `model.Chronos`.
			`negative_control_genes` (`None` or iterable): array-like of genes not expected to produce a viability phenotype. 
					If not included, negative_control_sgrnas must be passed.
					If 
			`negative_control_sgrnas` (`None` or `dict` of iterable): Needed if `negative_control_genes` not included. A per-library
					list of targeting sgRNAs not expected to produce a viability phenotype. See `model.Chronos`.
			`keep_models` (`bool`): if true, Chronos models trained in `.compare_conditions` will be retained.
			`intermediate_print_to` (`str` or callable or `None`): where to print ordinary messages from the Comparator. Default is `stdout`. 
					Pass a file path to write to the file or `None` to skip these messages.
			`print_to` (`str` or callable or `None`): where to print ordinary messages from child Chronos models. Default is `stdout`. 
					Pass a file path to write to the file or `None` to skip these messages.
			Additional keyword arguments will be passed to `model.Chronos` when training the models.
		'''
		self.printer = StdoutRedirector(intermediate_print_to)
		check_condition_map(condition_map)
		check_inputs(readcounts, guide_gene_map, condition_map)

		no_negative_control_genes = False
		try:
			negative_control_genes = list(negative_control_genes)
		except Exception:
			pass

		if not negative_control_genes:
			no_negative_control_genes = True
		else:
			negative_control_genes = sorted(set(negative_control_genes) & set.union(*[set(v.gene) for v in guide_gene_map.values()]))
			if not len(negative_control_genes):
				raise ValueError("negative_control_genes not present in any library's genes. Sample: \n%r" % negative_control_genes[:5])
		if not negative_control_sgrnas:
			if no_negative_control_genes:
				raise ValueError("one of `negative_control_genes` or `negative_control_sgrnas` must be specified")
			negative_control_sgrnas = {library: val.query("gene in %r" % list(negative_control_genes)).sgrna
			for library, val in guide_gene_map.items()}

		self.negative_control_genes = negative_control_genes
		self.negative_control_sgrnas = negative_control_sgrnas

		self.printer.print("checking for high negative control correlation between replicates in the same condition")
		self.excess_correlation_warning = check_for_excess_correlation(readcounts, condition_map, negative_control_sgrnas)
		if negative_control_genes is None:
			negative_control_genes = []
		if self.excess_correlation_warning and len(negative_control_genes) < 90:
			raise RuntimeError("The biological replicates are not independent. A good set of at least 90 negative control \
genes must be passed to check p-value calibration after compare_conditions is run. Num. negative control genes available: %d" % len(negative_control_genes))

		self.readcounts = readcounts
		self.condition_map = Chronos._make_pdna_unique(condition_map, readcounts)
		for key, val in self.condition_map.items():
			assign_condition_replicate_ID(val)
		self.guide_gene_map = guide_gene_map

		self.print_to = print_to
		self.keep_models = keep_models
		self.kwargs = kwargs
		self.keys = sorted(self.readcounts.keys())
		self._scratch_dir = None


	def _get_scratch_dir(self):
		'''
		Directory used to hand the trained undistinguished model over to the distinguished and
		permuted models. Needed for safety with concurrent runs in the same directory.
		'''
		if self._scratch_dir is None:
			self._scratch_dir = mkdtemp(prefix="chronos_compare_")
		return self._scratch_dir


	def _clear_scratch_dir(self):
		if self._scratch_dir is not None:
			rmtree(self._scratch_dir, ignore_errors=True)
			self._scratch_dir = None


	def _check_condition_pair(self,  condition_pair):
		if condition_pair is None:
			condition_pairs = {key: 
				sorted(set(self.condition_map[self.keys[0]].condition.dropna().unique()))
				for key in self.keys
			}
			if not all([
				all([
					condition_pairs[key][i] == condition_pairs[self.keys[0]][i]
					for i in range(2)
				]) and (len(condition_pairs[key]) == 2)
				for key in self.keys
			]):
				raise ValueError("if `condition_pair` is not provided, all the maps in \
`condition_map` must have exactly 2 nonnull unique values, and they must be the same in\
every map.")

			condition_pair = condition_pairs[self.keys[0]]
		if len(condition_pair) != 2:
			raise ValueError("if `condition_pair` is provided, it must have length 2.")

		return condition_pair


	def compare_conditions(self, condition_pair=None,
		allow_reversed_permutations=False,
			max_null_iterations=2,
			gene_readcount_total_bin_quantiles=[.05],
			fdr_method="FDR_TSBH", nepochs=301, 
			genpareto_kwargs={},
				**kwargs):
		'''
		Generate a table with the significance of differences in gene effect between two conditions.
		First, a model is trained with no distinction between the different conditions.
		Then, a model is trained distinguishing the different conditions by modifying `cell_line_name`
		in the `sequence_map`s. 
		Finally, the condition labels are permuted within each cell line and a new model is trained
		per permutation. Each cell line in each permutation is required to have a unique permutation
		with respect to the same line in the other permutations, so the total number of permutations
		is limited by the cell line with the fewest replicates. 
		P-values are estimated per gene and cell line  using the distribution of differences between 
		conditions in the models. Permutation results from different genes are grouped together according
		to total reads informing the estimate to increase statistical power. 
		Parameters:
			`condition_pair` (iteranble of len 2 or None): the two conditions to be compared. If None,
				requires that `self.condition_map` have only two uniue conditions for non-pDNA entries.
				Differences will be reported as the second condition minus the first.
			`gene_readcount_total_bin_quantiles`: gene effect estimates for genes informed by few total reads are
				noisier than those with abundant reads. Therefore, when calculating p-values, genes
				will be binned by total readcounts according to the quantiles passed here. 0 and 1 are assumed.
				 More bins improves control of false discovery in common essential genes, but at the price of raising the minimum 
				achievable p value (which is 1/number of samples in the null distribution).
			`allow_reversed-permutations` (`bool`): whether to allow permutations that are 
				mirror images of each other - e.g. if ['A', 'B', 'A', 'B'] is one permutation, whether 
				to also include ['B', 'A', 'B', 'A']. Mirror image permutations will cause Chronos to
				estimate highly similar gene effects, just with the condition labels swapped. Thus,
				the permutations are no longer independently distributed. 
			`max_null_permutations` (`int`): limits the number of permutations used in the null, useful
				in the case that there are many replicates in each cell line.
			`gene_readcount_total_bin_quantiles` (iterable of floats in [0, 1]): genes will be binned
				according to their total readcounts summed over all guides and replicates. Each
				bin will be fit separately. This reduces noise in genes with low evidence
				from preventing significance in genes with better evidence. The values give
				the quantile boundaries of the bins.
			`fdr_method`: which method to use with statsmodels multiplecorrections
			`n_epochs`: how long to train each model
			`genpareto_kwargs` (`dict`): Controls when to accept a generalized Pareto model
				for the tail of the likelihoods. See `loglikelihood_p` for these arguments.
			`kwargs`: additional keyword arguments are passed to `Chronos.train`.
		Returns:
			`statistics` (`pd.DataFrame`): A dataframe with the columns:
				`cell_line_name`: the cell line name from `condition_map`
				`gene`: the gene from `guide_gene_map`
				`gene_effect_undistinguished`: the Chronos-estimated gene effect found by treating sequences 
					of the same cell line in every condition as a replicate, i.e. not distinguishing on 
					condition.
				`gene_effect_in_<condition1>`, `gene_effect_in_<condition2>`: Chronos estimates of the gene 
					effect for the line treating each condition as a different cell line.
				`gene_effect_difference`: `gene_effect_in_<condition2>` - `gene_effect_in_<condition1>`
				`permuted_gene_effect_in_<condition>_<min, max or mean>`: the results from the permutation
					tests, with the min, max or mean taken over the different permutations.
				`permuted_difference_sd`: The standard deviation of the difference seen between conditions over 
					the different permutations.
				`permuted_difference_extreme`: the maximal absolute difference seen between conditions over the
					the different permutations.
				`observed_statistic`: the value calculated by applying `comparison_statistic` to the
					`comparison_effect`s in the undistinguished, condition1, and condition2 cases.
				`pval`: the empirical p-value that the observed statistic did not arise from the distribution of
					statistics seen with the permuted condition labels.
				`FDR`: Benjamini-Hochberg estimate from the `pval` distribution (within readcount bins)
				`permuted_<min, max, mean>_statistic`: as `observed_statistic`, summarized over the permutations

		'''

		condition_pair = self._check_condition_pair(condition_pair)

		self.undistinguished_map = {key: filter_sequence_map_by_condition(
			self.condition_map[key], condition_pair)
			for key in self.keys
		}
		for val in self.undistinguished_map.values():
			val["true_cell_line_name"] = val["cell_line_name"].copy()

		self.retained_readcounts = {
			key:self.readcounts[key].loc[self.undistinguished_map[key].sequence_ID]
			for key in self.keys
		}
		self.compared_lines = sorted(set.union(*[set(v.cell_line_name) for v in self.undistinguished_map.values()]))
		self.compared_lines.remove("pDNA")

		self.readcount_gene_totals = self.get_readcount_gene_totals(
			self.retained_readcounts, self.condition_map, self.guide_gene_map
		)

		try:
			self.printer.print("training model without conditions distinguished")
			self.undistinguished_likelihood = self.get_undistinguished_results(
				self.undistinguished_map, nepochs, **kwargs
			)

			self.printer.print("training model with conditions distinguished")
			self.distinguished_map, self.distinguished_likelihood, \
				distinguished_gene_effect = self.get_distinguished_results(
				self.undistinguished_map, condition_pair, nepochs,
				 **kwargs
			)

			self.printer.print("training models with permuted conditions")
			self.permuted_maps, self.permuted_likelihoods, \
				permuted_gene_effects = self.get_permuted_results(
				max_null_iterations, self.undistinguished_map, condition_pair,
				allow_reversed_permutations, nepochs,
				**kwargs
			)
		finally:
			self._clear_scratch_dir()

		gene_effect_in_alt, gene_effect_in_baseline, gene_effect_difference = self.get_gene_effect_difference(
			distinguished_gene_effect, condition_pair
		)


		gene_effect_annotations = {
			"gene_effect_in_%s" % condition_pair[0]: gene_effect_in_baseline.stack(),
			"gene_effect_in_%s" % condition_pair[1]: gene_effect_in_alt.stack(),
			"gene_effect_difference": gene_effect_difference.stack(),
		}
		for i, permuted_effect in enumerate(permuted_gene_effects):
			ge_alt, ge_baseline, ge_diff = self.get_gene_effect_difference(permuted_effect, condition_pair)
			gene_effect_annotations["gene_effect_difference_permutation_%i" % i] = ge_diff.stack()

		gene_effect_annotations = pd.DataFrame(gene_effect_annotations).reset_index()
		gene_effect_annotations.rename(columns={
			gene_effect_annotations.columns[0]: "cell_line_name",
			gene_effect_annotations.columns[1]: "gene",
		}, inplace=True)


		self.printer.print("calculating empirical significance")
		significance, genpareto_fit_properties = self.get_significance(
			gene_readcount_total_bin_quantiles,
			self.readcount_gene_totals,
			self.distinguished_map,
			self.compared_lines,
			self.undistinguished_likelihood,
			self.distinguished_likelihood, 
			self.permuted_likelihoods,
			**genpareto_kwargs
		)
		significance_groups = significance.groupby("cell_line_name")
		fdrs = []
		adjusted_pvals = []
		for line, group in significance_groups:
			group = group.dropna(subset="likelihood_pval")

			try:
				pvals = group.set_index("gene").likelihood_pval.dropna()
			except KeyError as e:
				print(group.columns)
				raise e

			if self.negative_control_genes:

				calibrated = check_calibration(
					group.query(
						"gene in %r" % list(self.negative_control_genes)
					)["likelihood_pval"]
				)

				if not calibrated:
					warn("p-values are not calibrated for negative controls. \
Learning adjustment (but not guaranteed)")

					try:

						pvals = adjust_p_values(
							pvals, 
							self.negative_control_genes
						)
						adjusted_pvals.append(pd.DataFrame({
							"neg_control_adjusted_pval": pvals.values,
							"cell_line_name": line,
							"gene": pvals.index
						}))

					except PNotCorrectedError as e:
						warn(str(e) + f"\n. P-value correction for {line} failed and FDR is based on uncorrected \
p-values for this cell line. FDRs may be optimistic or pessimistic.")

			for readcount_bin, subgroup in group.groupby("readcount_bin"):
				pvals_bin = pvals[pvals.index.isin(subgroup.gene.unique())]
				fdrs.append(pd.DataFrame({
					"likelihood_fdr": multipletests(pvals_bin, .05, method=fdr_method)[1],
					"cell_line_name": line,
					"gene": pvals_bin.index
				}))

		fdrs = pd.concat(fdrs)

		statistics = gene_effect_annotations\
			.merge(significance, on=["cell_line_name", "gene"], how="outer")\
			.merge(fdrs, on=["cell_line_name", "gene"], how="outer")

		if adjusted_pvals:
			adjusted_pvals = pd.concat(adjusted_pvals)
			statistics = statistics.merge(
				adjusted_pvals, on=["cell_line_name", "gene"], how="left"
			)

		return statistics, genpareto_fit_properties


	def get_readcount_gene_totals(self, retained_readcounts, condition_map, guide_gene_map):
		pdna_seqs = {key: condition_map[key].query("cell_line_name == 'pDNA'").sequence_ID
					for key in condition_map}
		readcount_gene_totals = sum_collapse_dataframes([
			retained_readcounts[key]\
						.drop(pdna_seqs[key], axis=0, errors="ignore")\
						.T.groupby(guide_gene_map[key].set_index("sgrna")["gene"])\
						.sum().T\
						.median(axis=0)
			for key in self.keys
		])
		return readcount_gene_totals


	def get_undistinguished_results(self, undistinguished_map, nepochs, **kwargs):
		undistinguished_model = Chronos(
			readcounts=self.retained_readcounts,
			sequence_map=undistinguished_map,
			guide_gene_map=self.guide_gene_map,
			negative_control_sgrnas=self.negative_control_sgrnas,
			 use_line_mean_as_reference=np.inf,
			 print_to=self.print_to,
			**self.kwargs
		)
		undistinguished_model.train(nepochs, **kwargs)
		undistinguished_model.save(self._get_scratch_dir(), overwrite=True)
		likelihood = cell_line_log_likelihood(undistinguished_model, undistinguished_map)
		if self.keep_models:
			self.undistinguished_model = undistinguished_model
		else:
			del undistinguished_model
		return likelihood


	def get_distinguished_results(self, undistinguished_map, condition_pair,
			 nepochs, **kwargs
		):

		distinguished_map = {key:
			create_condition_sequence_map(undistinguished_map[key], condition_pair)
			for key in self.keys}


		distinguished_model = Chronos(
			readcounts=self.retained_readcounts,
			sequence_map=distinguished_map,
			guide_gene_map=self.guide_gene_map,
			negative_control_sgrnas=self.negative_control_sgrnas,
			 use_line_mean_as_reference=np.inf,
			 pretrained=True,
			 constrained_mean=True,
			 print_to=self.print_to,
			**self.kwargs
		)
		distinguished_model.import_model(self._get_scratch_dir())
		growth_rate = pd.read_csv(os.path.join(self._get_scratch_dir(), "growth_rate.csv"))

		growth_rate = _remap_growth_rates(growth_rate, undistinguished_map,
			distinguished_map)

		distinguished_model.growth_rate = growth_rate
		distinguished_model.train(nepochs, ge_only=nepochs, **kwargs)

		distinguished_gene_effect = distinguished_model.gene_effect
		distinguished_likelihood = cell_line_log_likelihood(distinguished_model, distinguished_map)
		if self.keep_models:
			self.distinguished_model = distinguished_model
		else:
			del distinguished_model

		return distinguished_map, distinguished_likelihood, distinguished_gene_effect


	def get_permuted_results(self, max_null_iterations, undistinguished_map, condition_pair, 
			allow_reversed_permutations, nepochs, **kwargs
		):

		permuted_maps = {
			key: create_permuted_sequence_maps(undistinguished_map[key], condition_pair, 
												allow_reversed_permutations)
			for key in self.keys
		}
		min_permutations = min([len(permuted_maps[key]) for key in self.keys])
		permuted_maps = [{key: permuted_maps[key][i] for key in self.keys} for i in range(min_permutations)]
		out = []
		permuted_gene_effects = []
		counts = 0
		self.permuted_models = []
		for i, permuted_map in enumerate(permuted_maps):
			self.printer.print('\trandom iteration %i' %i)
				
			permuted_model = Chronos(readcounts=self.retained_readcounts, 
								 sequence_map=permuted_map,
								 guide_gene_map=self.guide_gene_map,
								 negative_control_sgrnas=self.negative_control_sgrnas,
								  use_line_mean_as_reference=np.inf,
								  pretrained=True,
								  constrained_mean=True,
								  print_to=self.print_to,
								**self.kwargs
								)
			permuted_model.import_model(self._get_scratch_dir())

			growth_rate = pd.read_csv(os.path.join(self._get_scratch_dir(), "growth_rate.csv"))
			permuted_model.growth_rate = _remap_growth_rates(growth_rate, undistinguished_map,
				permuted_map)

			permuted_model.train(nepochs, ge_only=nepochs, **kwargs)

			out.append(cell_line_log_likelihood(permuted_model, permuted_map))
			permuted_gene_effects.append(permuted_model.gene_effect)
			if self.keep_models:
				self.permuted_models.append(permuted_model)
			else:
				del permuted_model

			counts += 1
			if counts == max_null_iterations:
				break

		if not self.keep_models:
			del self.permuted_models
		return permuted_maps, out, permuted_gene_effects


	def get_gene_effect_difference(self, distinguished_gene_effect, condition_pair):
		gene_effect_in_alt = distinguished_gene_effect.loc[[
			'%s__in__%s' % (line, condition_pair[1])
			for line in self.compared_lines
		]]
		gene_effect_in_alt.set_index(np.array(self.compared_lines), inplace=True)

		gene_effect_in_baseline = distinguished_gene_effect.loc[[
			'%s__in__%s' % (line, condition_pair[0])
			for line in self.compared_lines
		]]
		gene_effect_in_baseline.set_index(np.array(self.compared_lines), inplace=True)

		gene_effect_difference = gene_effect_in_alt - gene_effect_in_baseline

		return gene_effect_in_alt, gene_effect_in_baseline, gene_effect_difference


	def get_significance(self, 
			gene_readcount_total_bin_quantiles, 
			readcount_gene_totals,
			distinguished_map, 
			compared_lines,
			undistinguished_likelihood, 
			distinguished_likelihood, 
			permuted_likelihoods, 
			noise_fraction=.2, 
			test_frac=.5, 
			delta=2, 
			conf=.95, 
			alpha_rank=50, 
			accept_underpowered=True, 
			accept_indeterminate=True,
			min_samples_for_genpareto=300, 
			seed=None,
			additional_annotations={}
		):
		'''
		get estimated p-values, FDRs, and other properties using the log likelihoods
		'''

		bins = readcount_gene_totals.quantile([0] + list(gene_readcount_total_bin_quantiles) + [1])
		bins[0.0] = -1
		bins[1.0] *= 1.05
		bins = pd.cut(readcount_gene_totals, bins)
		try:
			_ = sorted(bins.unique())
		except:
			raise ValueError("readcounts improperly binned, bins: %r" % bins.unique())

		out = []
		bin_assignments = []
		genpareto_fit_df = pd.DataFrame(
			columns=["certification", "alpha", "min_delta", "u0", "disagreement", "bias",
			"p_u0", "xi", "sigma"]
		)

		for line in compared_lines:
			if line == 'pDNA':
				continue


			for bin in sorted(bins.unique()):
				genes = bins.loc[lambda x: x==bin].index

				if len(genes) < 500:
					warn("Only %i genes in one of your bins. This will limit the minimum achievable p-value \
	. If you have a sub-genome library, considering changing `gene_readcount_total_bin_quantiles` so there are \
	more genes in each bin." % (len(genes)))

				null = pd.concat([v.loc[line, genes] - undistinguished_likelihood.loc[line, genes] 
					for v in permuted_likelihoods], ignore_index=True)
				observed = distinguished_likelihood.loc[line, genes] - undistinguished_likelihood.loc[line, genes]

				p, genpareto_fit_properties = loglikelihood_p(
					observed.values, null.values, noise_fraction, test_frac, 
					delta, conf, alpha_rank, accept_underpowered, accept_indeterminate,
					min_samples_for_genpareto,  
					seed
				)

				out.append(pd.DataFrame({
					"likelihood": distinguished_likelihood.loc[line, genes],
					"likelihood_undistinguished": undistinguished_likelihood.loc[line, genes],
					"likelihood_permutation_0": permuted_likelihoods[0].loc[line, genes],
					"likelihood_permutation_1": permuted_likelihoods[1].loc[line, genes],
					"likelihood_pval": p, 
					"cell_line_name": line,
					"readcount_bin": bin
				}))

				genpareto_fit_df.loc[f"{line}:{bin}"] = pd.Series(
					genpareto_fit_properties
				).dropna()

				for key, val in additional_annotations:
					if isinstance(val, pd.Series):
						try:
							val = val.loc[line].loc[genes]
						except IndexError:
							raise ValueError("additional annotation '%s' missing genes:\n%r" %
								(key, val))
					out[-1][key] = val

				out[-1].reset_index(inplace=True)
				out[-1].rename(columns={out[-1].columns[0]: "gene"}, inplace=True)
		return pd.concat(out, ignore_index=True), genpareto_fit_df


	def __del__(self):
		try:
			del self.undistinguished_model, self.distinguished_model
			for model in list(self.permuted_models):
				del model
		except AttributeError:
			pass


def get_consensus_difference_statistics(comparison_statistics):
	'''
	Get a single p-value for each gene that its viability is different between the conditions,
	summarized over cell lines. It does this by first checking the mean difference for each gene.
	Cell lines where the difference for that gene has the opposite sign have their change in
	likelihood forced to be negative. Then, the likelihood changes are summed across cell lines
	for both the real comparison and the permuted label (null hypothesis) comparisons. P-values
	are calculated as in `compare_conditions`.

	Parameters:
		`comparison_statistics`: output of `compare_conditions`
	Returns:
		`consensus_statistics`: `pnadas.DataFrame` in the same format as `comparison_statistics`
	'''
	cs = comparison_statistics.copy()
	cs["likelihood_difference"] = cs["likelihood"] - cs["likelihood_undistinguished"]
	n_perms = len([s for s in cs.columns if s.startswith("likelihood_permutation")])
	for i in range(n_perms):
		cs["likelihood_difference_permutation_%i"%i] = cs["likelihood_permutation_%i" %i] - cs["likelihood_undistinguished"]
	
	def get_adjusted_likelihood(ge_diff_col, ll_diff_col):
		means = cs.groupby("gene")[ge_diff_col].mean()
		adjusted_ll_diffs = []
		for line, group in cs.groupby("cell_line_name"):
			adjusted_ll_diff = group[ll_diff_col].copy()
			mean_sign = np.sign(means.loc[group.gene].values)
			adjusted_ll_diff[mean_sign != np.sign(group[ge_diff_col])] = -np.abs(adjusted_ll_diff[mean_sign != np.sign(group[ge_diff_col])])
			adjusted_ll_diffs.append(adjusted_ll_diff)
		return pd.concat(adjusted_ll_diffs), means
	
	means = {}
	cs["adjusted_likelihood_difference"], means["mean_gene_effect_difference"] = get_adjusted_likelihood(
		"gene_effect_difference", 
		"likelihood_difference"
	)
	for i in range(n_perms):
		(
			cs["adjusted_likelihood_difference_permutation_%i" % i],
			means["mean_gene_effect_difference_permutation_%i" %i]) = get_adjusted_likelihood(
			"gene_effect_difference_permutation_%i" % i, 
			"likelihood_difference_permutation_%i" %i
		)
	

	null = pd.concat([
		cs["adjusted_likelihood_difference_permutation_%i" %i] 
		for i in range(n_perms)
	],ignore_index=True)

	cs['adjusted_p'] = empirical_pvalue_lognorm_extension(
		cs["adjusted_likelihood_difference"], 
		null, 
		direction=1
	)
	means["likelihood_p"] = cs\
		.groupby("gene")\
		["adjusted_p"]\
		.apply(lambda x: combine_pvalues(x)[1])

	mask = means["likelihood_p"].notnull()
	fdr = pd.Series(
		multipletests(means["likelihood_p"][mask].values, .05, method="FDR_TSBH")[1], 
		index=means["likelihood_p"].index[mask])
	means["likelihood_fdr"] = fdr
	means = pd.DataFrame(means)
	return means


################################################################
# P R O B A B I L I T I E S
################################################################


### Infer class probabilities
class MixFitEmpirical:
	'''
	fit a 1D mixture model with a set of known `n` functions using E-M optimization
	Attributes:
		`lambda` (`numpy.ndarray` of `float`, shape (n)): the mixing fraction. `lambda[i]` is the 
			current estimate of the total fraction of samples belonging to distribution i.
		`n` (`int`): the number of distributions being fit
		`densities` (iterable of `function`, shape = (n)): functions accepting 1D arrays 
			and returning the same shape array with a probability density estimate for each value
		`data` (`numpy.ndarray` of `float`, shape = (k)): 1D array of observed values
		`p` (`numpy.ndarray` of `float` in [0, 1], shape = (n, k)): `p[i, j]` is the probability 
			density `density[n](data[k])`
		`q` (`numpy.ndarray` of `float` in [0, 1], shape = (n, k)): the posterior probability that
			a point in `data` belongs to a distribution. 
		`likelihood` (`float`): The mean log likelhood of the `data` given `p` and `q`.
	'''
	def __init__(self, densities, data, initial_lambdas=None):
		'''
		Parameters:
			`densities`: iterable of normalized functions R^N -> P^N
			`data`: 1D array of points
			`initial_lambdas`: iterable with same length as densities giving initial guesses
				for size of each component. Must sum to 1.
		'''
		if initial_lambdas is None:
			initial_lambdas = [1.0/len(densities) for d in densities]
		if sum(pd.isnull(densities)) > 0:
			e = 'Error: %i null values in data\n%r' %(sum(pd.isnull(densities)), data)
			raise ValueError(e)
		self.lambdas = np.array(initial_lambdas)
		self.n = len(initial_lambdas)
		self.densities = list(densities)
		self.data = np.array(data)
		assert sum(initial_lambdas) == 1, 'Invalid mixing sizes'
		self.q = 1.0/(len(self.densities)) * np.ones((len(self.densities), len(self.data)))
		self.p = np.stack([d(self.data) for d in densities])
		self.likelihood = self.get_likelihood()

	
	def fit(self, tol=1e-7, maxit=1000, lambda_lock=False, verbose=False):
		'''
		Maximize likelihood. 
		Parameters:
			`tol` (`float`): consider converged when the fractional increase in likelihood 
				is less than this.
			`maxit` (`int`): maximum iterations to attempt for convergence.
			`lambda_lock` (`bool`): whether to update the total proportions
				of the observations estimated to be generated from each of the distributions.
				if `False`, will use the `initial_lambdas`. 
		'''
		for i in range(maxit):

			#E step
			for k in range(self.n):
				self.q[k, :] = self.lambdas[k]*self.p[k, :]
			if any(np.sum(self.q, axis=0) == 0):
				loc = np.argwhere(np.sum(self.q, axis=0) == 0).ravel()
				bad_points = self.data[loc]
				e = 'All component densities invalid for indices %r\ndata there: %r\ndensities there: %r)' %(
					loc, bad_points, [d(bad_points) for d in self.densities])
				raise ValueError(e)
			self.q /= np.sum(self.q, axis=0)

			#M step
			if not lambda_lock:
				self.lambdas = np.mean(self.q, axis=1)
				self.lambdas /= sum(self.lambdas)

			new_likelihood = self.get_likelihood()
			last_change = (new_likelihood - self.likelihood)/self.likelihood
			self.likelihood = new_likelihood
			if 0 <= last_change < tol:
				if verbose:
					print("Converged at likelihood %f with %i iterations" %(new_likelihood, i))
				break
		if (i >= maxit - 1) and verbose:
			print("Reached max iterations %i with likelihood %f, last change %f" %(
				maxit, self.likelihood, last_change))
		
	def get_likelihood(self):
		'''get the mean log likelihood of `self.data` given current posterior `self.q`.'''
		out = np.mean(
			np.log(
				np.sum(
					self.q*self.p,
					axis=0
				)
			)
		)

		if np.isnan(out):
			e = "Null cost. %i nulls and %i negatives in q values, %i nulls and %i negatives in densities,\
%i nulls and %i negatives in p values\
			\n%r\n%r" %(
				np.sum(np.isnan(self.q)),
				np.sum(self.q < 0),
				np.sum(np.isnan(np.stack([d(self.data) for d in self.densities]))),
				np.sum(np.stack([d(self.data) for d in self.densities]) < 0),
				np.sum(np.isnan(self.p)),
				np.sum(self.p < 0),
				np.argwhere(np.isnan(self.p)),
				self.p.shape
				)
			raise ValueError(e)
		return out

	def component_probability(self, points, component):
		'''get probability that data at specified points belongs to the given component (int)'''
		return self.lambdas[component] * self.densities[component](points) / sum([
			l*d(points) for l, d in zip(self.lambdas, self.densities)
			])
			

def probability_2class(component0_points, component1_points, all_points,
			   smoothing='scott',
			   right_kernel_threshold=1, left_kernel_threshold=-3,
			   maxit=500, lambda_lock=False, verbose=False, **mixfit_kwargs, 
			   ):
	'''
	Estimates the distributions of component0_points and component1_points using a gaussian 
	kernel, then assigns each of all_points a probability of belonging to the component1 
	distribution, assuming that the probability is a logit function of the value.
	Note that this is NOT a p-value: P(component1) = 1 - P(component0). 
	Component1 is always assumed to have more negative values than component0. 
	Parameters:
		component0_points (1D iterable): a set of values sampled from component0
		component1_points (1D iterable): a set of values sampled from component1
		all_points (`pandas.Series`): the values to be assigned probabilities
		smoothing (default "scott"): argument passed to `scipy.stats.guassian_kde` for 
			`bw_method`
		right_kernel_threshold (`float`): the value above which component0 will always have
			a small, finite density
		left_kernel_threshold (`float`): the value below which component1 will always have a 
			small, finite density
		maxit (`int`): maximum iterations to allow when calling `MixFitEmperical.fit`.
		lambda_lock (`bool`): whether to lock the estimated fraction of all points coming
			from component1 to the initial guess (50%, unless `lambda` is passed)
		verbose (`bool`): whether to print statements from `MixFitEmpirical.fit`
		Additional keyword args are passed to the `MixFitEmpirical` constructor.
	Returns:
		1D `numpy.ndarray` of the posterior probability of being generated from component1 vs 
			component0 for each point in all_points.
	'''
	#estimate density
	estimates = [
			gaussian_kde(component0_points, bw_method=smoothing),
			gaussian_kde(component1_points, bw_method=smoothing)
		]
	grid = np.arange(min(all_points) - .02, max(all_points) + .02, .01)
	estimates = [e(grid) for e in estimates]
	
	#this step copes with the fact that scipy's gaussian KDE often decays to true 0 in the far tails, leading to
	#undefined behavior in the mixture model
	if right_kernel_threshold is not None:
		estimates[0][np.logical_and(grid > right_kernel_threshold, estimates[0] <1e-16)] = 1e-16
	if left_kernel_threshold is not None:
		estimates[1][np.logical_and(grid < left_kernel_threshold, estimates[1] <1e-16)] = 1e-16

	#create density functions using interpolation (these are faster than calling KDE on grid)
	densities = [interp1d(grid, e) for e in estimates]
	
	#infer probabilities
	fitter = MixFitEmpirical(densities, all_points, **mixfit_kwargs)
	fitter.fit(maxit=maxit, lambda_lock=lambda_lock, verbose=verbose)

	# make monotonic
	probs = fitter.q[1]
	low_bound = component1_points.min()
	high_bound = component0_points.median()
	mask = (all_points>= low_bound) & (all_points <= high_bound)
	exog = sm.add_constant(all_points.values.reshape((-1, 1)))
	binomial_model = sm.GLM(probs[mask.values], exog[mask.values], family=sm.families.Binomial())
	binomial_results = binomial_model.fit()
	monotonic_probs = binomial_results.predict(exog)
	return monotonic_probs


def _check_controls(control_cols, control_matrix, gene_effect, label):
	var_name = '_'.join(label.split(' '))
	if control_cols is None and control_matrix is None:
		raise ValueError("One of `{0}` or `{0}_matrix` must be specified".format(var_name))
	if not (control_cols is None or control_matrix is None):
		raise ValueError("You can only specify one of `{0}` or `{0}_matrix` ".format(var_name))

	if control_matrix is None:
		if hasattr(control_cols, "shape"):
			if len(control_cols.shape) > 1:
				raise ValueError(f'if passed, {label} must be a 1D iterable')
		missing = list(set(control_cols) - set(gene_effect.columns))
		if missing:
			warn("Not all %s found in the gene effect columns: %r" % (label, missing[:5]))
		control_cols = sorted(set(control_cols) & set(gene_effect.columns))
		if len(control_cols) < 10:
			raise ValueError("Less than 10 (%i) %s found in gene effect" % (len(control_cols), label))
		if len(control_cols) < 200:
			warn("Less than 200 (%i) %s found in gene effect, inference may be low quality" % 
				(len(control_cols), label))
		control_matrix = pd.DataFrame(False, index=gene_effect.index, columns=gene_effect.columns)
		control_matrix[control_cols] = True

	missing = set(gene_effect.index) - set(control_matrix.index)
	if len(missing) == len(gene_effect):
		raise ValueError("The index of `gene_effect` does not match `%s_matrix`." % (
			var_name))
	if len(missing):
		raise ValueError("Not all screens in `gene_effect` have entries in `%s_matrix`" % var_name)
	if not len(set(control_matrix.columns) & set(gene_effect.columns)):
		raise ValueError("None of the genes in `gene_effect` are in the columns of `%s_matrix`"
			% (var_name))
	control_matrix = control_matrix\
		.loc[gene_effect.index]\
		.reindex(columns=gene_effect.columns)\
		.fillna(False)
	if (control_matrix.sum(axis=1) < 200).any():
		warn("Less than 200 %s in some screens, inference may be low quality" % label)
	return control_matrix


def get_probability_dependent(gene_effect, 
	negative_controls=None, positive_controls=None, 
	negative_control_matrix=None, positive_control_matrix=None,
	verbose=False,
	**kwargs):
	'''
	Generates a matrix of the same dimensions as `gene_effect` where the values are probabilites
	that the corresponding `gene_effect` value was drawn from the distribution of 
	`positive_controls` rather than `negative_controls`.
	Parameters:
		`gene_effect` (`pandas.DataFrame`): Chronos gene effect estimates
		`negative_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as 
			negative controls (null distribution). `negative_controls` or 
			`negative_control_matrix` must be specified.
		`positive_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as
			positive_controls (high confidence that loss leads to loss of viability).
			One of `positive_controls` or `positive_control_matrix` must be specified.
		`negative_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a negative control
			in the cell line in question. This is useful when the controls differ betweeen different
			cell lines, such as when using unexpressed genes as the negative controls. 
		`positive_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a positive control
			in the cell line in question.  
		Other arguments passed to `probability_2class`
	Returns:
		`pandas.DataFrame` containing the probability estimates.
	'''
	

	positive_control_matrix = _check_controls(positive_controls, positive_control_matrix,
								gene_effect,  "positive controls")
	negative_control_matrix = _check_controls(negative_controls, negative_control_matrix,
								gene_effect,  "negative controls")

	# centers negative controls at 0. This is important for the tail kernels to behave correctly.
	gene_effect = gene_effect - np.nanmean(gene_effect.mask(~negative_control_matrix))

	def row_probability(x):
		component0 = x[negative_control_matrix.loc[x.name]].dropna()
		component1 = x[positive_control_matrix.loc[x.name]].dropna()
		if not len(component0):
			raise KeyError("no non-null negative controls found for %s" % x.name)
		if not len(component1):
			raise KeyError("no non-null positive controls found for %s" % x.name)
		return probability_2class(
			component0, 
			component1, 
			x.dropna(), 
			verbose=verbose,
			**kwargs
		)

	return pd.DataFrame({
		ind: pd.Series(row_probability(row), index=row.dropna().index)
		for ind, row in tqdm(gene_effect.iterrows(), total=len(gene_effect))
	}).reindex(index=gene_effect.columns).T


def get_fdr_from_probabilities(probabilities):
	'''Computes the (Bayesian) false discovery rates from the probability of dependency matrix 
	`probabilities` and returns it as a matrix. The FDR for an observation in a given
	cell line  is 1 - the mean probability of dependency for all observations
	in that cell line with equal or more negative gene effect than the observation.
	''' 
	out = {}
	for ind, row in probabilities.iterrows():
		row = row.sort_values(ascending=False)
		out[ind] = pd.Series(
			np.cumsum(1-row.fillna(0)) / np.cumsum(row.notnull()),
			index=row.index
		)
	return pd.DataFrame(out).reindex(index=probabilities.columns).T


def get_pvalue_dependent(gene_effect, negative_controls=None, negative_control_matrix=None):
	'''
	Generates a matrix of the same dimensions as `gene_effect` where the values are p-values
	for the null hypothesis that the true corresponding `gene_effect` value represented no viability
	effect against the alternative that there was loss of viability using `empirical_pvalue`.
	Parameters:
		`gene_effect` (`pandas.DataFrame`): Chronos gene effect estimates
		`negative_controls` (iterable of `str` or `None`): the genes in `gene_effect` to treat as 
			negative controls (null distribution). `negative_controls` or 
			`negative_control_matrix` must be specified.
		`negative_control_matrix` (`pandas.DataFrame` or `None`): a matrix of boolean values
			matching `gene_effect` where `True` indicates that that gene is a negative control
			in the cell line in question. This is useful when the controls differ betweeen different
			cell lines, such as when using unexpressed genes as the negative controls. 
	Returns:
		`pandas.DataFrame` containing the p-value estimates.
	'''
	negative_control_matrix = _check_controls(negative_controls, negative_control_matrix,
		gene_effect, "negative_controls")
	return pd.DataFrame({
		ind: empirical_pvalue(row, row[negative_control_matrix.loc[ind]])
		for ind, row in gene_effect.iterrows()
	}).T


def get_fdr_from_pvalues(pvalues, method="FDR_TSBH"):
	'''Computes the Benjamini-Hochberg corrected p-values (frequentist false discovery rates)
	from the p-value matrix and returns it as a matrix. 
	FDRs are computed within individual cell lines (rows).
	''' 
	out = {}
	for ind, row in pvalues.iterrows():
		row = row.dropna()
		out[ind] = pd.Series(
			multipletests(row, .05, method=method)[1],
			index=row.index
		)
	return pd.DataFrame(out).reindex(index=pvalues.columns).T