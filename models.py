import numpy as np
import pandas as pd

from product_factors import product_factor_preds
from data import counts_and_avgs, log, load_data, load_no_name_clients, densify, densify2

GLOBAL_MEDIAN = 3

def reference(train, test, clients, products, data_name):
	return predict_reference(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def current(train, test, clients, products, data_name):
	return predict_current(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def predict_current(train, test, data_name, default):
	# factor_preds = product_factor_preds(train, test, data_name)

	print "building (client, depot, product) keys"
	triples = (lambda frame:
			frame.client_key.values.astype(np.int64) * (5000 * 600) 
			+ frame.product_key.values * 600 
			+ frame.depot_key.values)
	train_keys, test_keys = densify2(triples(train), triples(test))
	print len(train_keys), "train_keys", len(test_keys), "test_keys"

	print "finding log avg for triples in training"
	counts, means = counts_and_avgs(train_keys, train.log_sales.values)

	print "predictions"
	print "\t%d total:" % len(test)
	preds = means[test_keys]
	nans = np.isnan(preds)
	triple_pred_count = len(test) - np.count_nonzero(nans)
	print "\t%d from pair means" % triple_pred_count
	# preds[nans] = factor_preds[nans]
	# nans = np.isnan(preds)
	# factor_pred_count = len(test) - pair_pred_count - np.count_nonzero(nans)
	# print "\t%d from product factors" % factor_pred_count
	# nans = np.isnan(preds)
	median_count = np.count_nonzero(nans)
	preds[nans] = default
	print "\t%d from median" % default
	print "exp()"
	preds = np.exp(preds) - 1
	return preds, counts[test_keys]

def predict_reference(train, test, data_name, default):
	# # # DEBUG
	# train, test, _, _, _ = load_data()
	# data_name = "debug"
	# default = np.log(GLOBAL_MEDIAN + 1)

	# factor_preds = product_factor_preds(train, test, data_name)

	print "building (client, product) keys"
	train_pairs = train.client_key.values.astype(np.int64) * 5000 + train.product_key.values
	test_pairs = test.client_key.values.astype(np.int64) * 5000 + test.product_key.values
	train_pair_keys, test_pair_keys = densify2(train_pairs, test_pairs)
	print len(train_pair_keys), "train_pair_keys", len(test_pair_keys), "test_pair_keys"

	print "finding log avg for pairs in training"
	pair_counts, pair_means = counts_and_avgs(train_pair_keys, train.log_sales.values)

	print "predictions"
	print "\t%d total:" % len(test)
	preds = pair_means[test_pair_keys]
	nans = np.isnan(preds)
	pair_pred_count = len(test) - np.count_nonzero(nans)
	print "\t%d from pair means" % pair_pred_count
	# preds[nans] = factor_preds[nans]
	# nans = np.isnan(preds)
	# factor_pred_count = len(test) - pair_pred_count - np.count_nonzero(nans)
	# print "\t%d from product factors" % factor_pred_count
	# nans = np.isnan(preds)
	median_count = np.count_nonzero(nans)
	preds[nans] = default
	print "\t%d from median" % default
	print "exp()"
	preds = np.exp(preds) - 1
	return preds, pair_counts[test_pair_keys]


