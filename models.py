import numpy as np
import pandas as pd
import xgboost as xgb

from product_factors import *
from data import counts_and_avgs, log, load_data, load_no_name_clients, densify

GLOBAL_MEDIAN = 3.0

def reference(train, test, clients, products, data_name):
	return predict_reference(train, test, clients)

def current(train, test, clients, products, data_name):
	return predict_current(train, test, clients)

def predict_current(train, test, clients):
	print "current predictions"
	print "\t%d total:" % len(test)
	preds, counts = by_client_product(train, test)
	preds[np.isnan(preds)] = by_client_factor_vs_product(train, test[np.isnan(preds)])
	preds[np.isnan(preds)] = by_clientname_product(train, test[np.isnan(preds)], clients)
	preds[np.isnan(preds)] = by_median(test[np.isnan(preds)])
	return preds, counts

def predict_reference(train, test, clients):
	print "reference predictions"
	print "\t%d total:" % len(test)
	preds, counts = by_client_product(train, test)
	preds[np.isnan(preds)] = by_product_factor_vs_client(train, test[np.isnan(preds)])
	preds[np.isnan(preds)] = by_clientname_product(train, test[np.isnan(preds)], clients)
	preds[np.isnan(preds)] = by_median(test[np.isnan(preds)])
	return preds, counts

def by_key(train, test, key_fn):
	print "\tbuilding keys"
	train_keys, test_keys = densify(key_fn(train), key_fn(test))

	print "\tfinding log means"
	counts, means = counts_and_avgs(train_keys, train.log_sales.values)

	print "\tpredicting"
	preds = means[test_keys]
	print "\texp()"
	preds = np.exp(preds) - 1
	print "%d non-NaN" % np.count_nonzero(~np.isnan(preds))
	return preds, counts[test_keys]

def by_client_product(train, test):
	print "by (client, product)"
	return by_key(train, test, lambda frame: 
		frame.client_key.values.astype(np.int64) * 3000 
		+ frame.product_key.values)

def by_client_depot_product(train, test):
	print "by (client, depot, product)"
	return by_key(train, test, lambda frame: 
		frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

def by_median(test):
	print "by median: %d preds" % len(test)
	return np.ones(len(test), dtype=np.float32) * GLOBAL_MEDIAN

def by_clientname_product(train, test, clients):
	print "predictions on (client_name, product)"
	print "\tbuilding keys"
	train_client_key, test_client_key, client_client_key = densify(train.client_id.values, test.client_id.values, clients.index.values)
	
	print "\thashing client names"
	client_hashes = np.zeros(np.max(client_client_key) + 1, dtype=np.int64)
	for r, name in enumerate(clients.client_name):
		client_key = client_client_key[r]
		client_hashes[client_key] = hash(name)

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



