import numpy as np
import pandas as pd
import os, pickle

from data import counts_and_avgs, densify

def product_factor_vs_client_features(train, test):
	return avg_factor_features(train, test, 
		lambda frame: frame.client_key.values,
		lambda frame: frame.product_key.values)

def client_factor_vs_product_features(train, test):
	return avg_factor_features(train, test, 
		lambda frame: frame.product_key.values,
		lambda frame: frame.client_key.values)

def avg_factor_features(train, test, baseline_key_fn, group_key_fn):
	print "\tdense keys"
	train_base_keys, test_base_keys = densify(baseline_key_fn(train), baseline_key_fn(test))
	train_group_keys, test_group_keys = densify(group_key_fn(train), group_key_fn(test))

	print "\taverages for baselines"
	_, base_avgs = counts_and_avgs(train_base_keys, train.net_sales.values)

	print "\tbroadcasting to train"
	train_baselines = base_avgs[train_base_keys]

	print "\tfinding train factors"
	train_factors = train.net_sales.values / train_baselines
	train_factors[train_baselines == 0] = np.NaN

	nans = np.isnan(train_factors)
	print "\taveraging by group key (skipping %d NaNs)" % np.count_nonzero(nans)
	_, group_factors = counts_and_avgs(train_group_keys[~nans], train_factors[~nans])

	print "\tbroadcasting to test"
	test_baselines = base_avgs[test_base_keys]

	print "\tcomputing"
	return group_factors[test_group_keys] * test_baselines

# def by_avg_factor(train, test, baseline_key_fn, group_key_fn):
# 	print "\tdense keys"
# 	train_base_keys, test_base_keys = densify(baseline_key_fn(train), baseline_key_fn(test))
# 	train_group_keys, test_group_keys = densify(group_key_fn(train), group_key_fn(test))

# 	print "\taverages for baselines"
# 	_, base_avgs = counts_and_avgs(train_base_keys, train.net_sales.values)

# 	print "\tbroadcasting to train"
# 	train_baselines = base_avgs[train_base_keys]

# 	print "\tfinding train factors"
# 	train_factors = train.net_sales.values / train_baselines
# 	train_factors[train_baselines == 0] = np.NaN

# 	nans = np.isnan(train_factors)
# 	print "\taveraging by group key (skipping %d NaNs)" % np.count_nonzero(nans)
# 	_, group_factors = counts_and_avgs(train_group_keys[~nans], train_factors[~nans])

# 	print "\tbroadcasting to test"
# 	test_baselines = base_avgs[test_base_keys]
# 	test_factors = group_factors[test_group_keys] # this assumes all the test groups have been seen with non-NaN values

# 	print "\tcomputing"
# 	preds = test_factors * test_baselines
# 	preds[test_baselines == 0] = np.NaN
# 	print "%d non-NaN predictions" % np.count_nonzero(~np.isnan(preds))
# 	print "\texp()"
# 	return np.exp(preds) - 1
