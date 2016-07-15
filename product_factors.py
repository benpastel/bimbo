import numpy as np
import pandas as pd
import os, pickle

from data import counts_and_avgs, load_data

def hash_client_depot(client_key, depot_key):
	return client_key * 552 + depot_key

def product_factor_preds(train, test, data_name):
	path = "pickle/%s_product_factor_preds.pickle" % data_name
	if os.path.isfile(path):
		print "loading %s..." % path
		with open(path, 'r') as f:
			preds = pickle.load(f)
			if len(preds) != len(test):
				raise Exception("bad save file")
			return preds

	print "training product factors"
	print "\thashing"
	train_hashes = hash_client_depot(train.client_key, train.depot_key)
	# index: training row

	print "\taveraging by (client, depot)"
	_, baseline_avgs = counts_and_avgs(train_hashes, train.log_sales)
	# index: hash value

	print "\tbroadcasting to train"
	baselines_for_train = baseline_avgs[train_hashes]
	# index: training row
	# all non-NaN, but 3176 == 0

	print "\tfinding factors"
	factors = train.log_sales / baselines_for_train
	factors[baselines_for_train == 0] = np.NaN
	# index: training row

	print "\taveraging by product"
	_, product_factors = counts_and_avgs(train.product_key, factors)
	# index: product_key

	print "making test predictions"
	print "\thashing"
	test_hashes = hash_client_depot(test.client_key, test.depot_key)
	# index: test row

	print "\tbroadcasting"
	factors_for_test = product_factors[test.product_key]
	baselines_for_test = baseline_avgs[test_hashes]

	print "\tcomputing"
	preds = factors_for_test / baselines_for_test

	nans = np.count_nonzero(np.isnan(preds))
	print "made %d predictions and %d NaNs" % ((len(preds) - nans), nans)

	print "saving to", path
	with open(path, 'w') as f:
		pickle.dump(preds, f)
	return preds