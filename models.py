import numpy as np
import pandas as pd
import xgboost as xgb

from product_factors import product_factor_preds
from data import counts_and_avgs, log, load_data, load_no_name_clients, densify

GLOBAL_MEDIAN = 3.0

def reference(train, test, clients, products, data_name):
	return predict_reference(train, test)

def current(train, test, clients, products, data_name):
	return predict_current(train, test, clients)

def predict_current(train, test, clients):
	print "current predictions"
	print "\t%d total:" % len(test)
	preds, counts = by_clientname_product(train, test, clients)
	nans = np.isnan(preds)
	preds[nans] = by_median(test[nans])
	return preds, counts

def predict_reference(train, test):
	print "reference predictions"
	print "\t%d total:" % len(test)
	preds, counts = by_client_product(train, test)
	nans = np.isnan(preds)
	preds[nans] = by_median(test[nans])
	return preds, counts

def by_client_product(train, test):
	print "predictions on (client, product) log avg"
	print "\tbuilding keys"
	train_pairs = train.client_key.values.astype(np.int64) * 5000 + train.product_key.values
	test_pairs = test.client_key.values.astype(np.int64) * 5000 + test.product_key.values
	train_pair_keys, test_pair_keys = densify(train_pairs, test_pairs)

	print "\tfinding avgs"
	pair_counts, pair_means = counts_and_avgs(train_pair_keys, train.log_sales.values)

	print "\tpredicting"
	preds = pair_means[test_pair_keys]
	print "\texp()"
	preds = np.exp(preds) - 1
	print "\t%d non-NaN" % np.count_nonzero(~np.isnan(preds))
	return preds, pair_counts[test_pair_keys]

def by_median(test):
	print "\t%d preds from median" % len(test)
	return np.ones(len(test), dtype=np.float32) * GLOBAL_MEDIAN

def by_clientname_product(train, test, clients):
	print "predictions on (client_name, product)"
	print "\tbuilding keys"
	train_client_key, test_client_key, client_client_key = densify(train.client_id.values, test.client_id.values, clients.index.values)
	
	print "\thashing client names"
	# array of hashes indexed on client_key
	client_hashes = np.zeros(np.max(client_client_key) + 1, dtype=np.int64)

	for r, name in enumerate(clients.client_name):
		client_key = client_client_key[r]
		client_hashes[client_key] = hash(name)
	print "\tuniques before / after hashing: %d / %d" % (len(clients.client_name.unique()), len(np.unique(client_hashes)))

	def key(frame, frame_client_keys):
		return (
			client_hashes[frame_client_keys] * 3000
			+ frame.product_key.values)
	train_keys, test_keys = densify(key(train, train_client_key), key(test, test_client_key))
	print len(train_keys), "train_keys", len(test_keys), "test_keys"

	print "\tfinding means in training"
	counts, means = counts_and_avgs(train_keys, train.log_sales.values)

	print "\tpredicting"
	preds = means[test_keys]
	print "\texp()"
	preds = np.exp(preds) - 1
	print "\tmade %d non-NaN predictions" % np.count_nonzero(~np.isnan(preds))
	return preds, counts[test_keys]



