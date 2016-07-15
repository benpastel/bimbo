import numpy as np
import pandas as pd

from product_factors import product_factor_preds
from data import counts_and_avgs, log, load_data, load_no_name_clients

GLOBAL_MEDIAN = 3

def reference(train, test, clients, products, data_name):
	return predict_reference(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def current(train, test, clients, products, data_name):
	return predict_current(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def predict_current(train, test, data_name, default):
	factor_preds = product_factor_preds(train, test, data_name)

	# TODO compare vs. median
	print "finding log avgs by (client, depot, product)"
	print "\tcounts"
	triple_counts = train.groupby(["client_key", "depot_key", "product_key"]).log_avg.count()
	print "\tmeans"
	triple_means = train.groupby(["client_key", "depot_key", "product_key"]).log_avg.mean()

	print "predictions"
	simples = 0
	factors = 0
	medians = 0
	counts = np.zeros(len(test))
	preds = np.zeros(len(test))

	r = 0
	for c, p, d in test[["client_key", "product_key", "depot_key"]].itertuples(False):
		if (c, d, p) in triple_means:
			counts[r] = triple_counts[c, d, p]
			simples += 1
			preds[r] = triple_means[c, d, p]
			continue
		counts[r] = 0
		if not np.isnan(factor_preds[r]):
			factors += 1
			preds[r] = factor_preds[r]
		else:
			medians += 1
			preds[r] = default
		r += 1
	print "exp()"
	preds = np.exp(preds) - 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (simples, factors, medians)
	return preds, counts

def predict_reference(train, test, data_name, default):
	# # DEBUG
	# train, test, _, _, _ = load_data()
	# data_name = "debug"
	# default = np.log(GLOBAL_MEDIAN + 1)

	factor_preds = product_factor_preds(train, test, data_name)

	print "finding log avgs by (client, product)"
	pairs = train.groupby(["client_key", "product_key"])
	print "\tcounts"
	pair_counts = pairs.log_sales.count()
	print "\tmeans"
	pair_means = pairs.log_sales.mean()
	print "\tkeys"
	pair_keys = set(pairs.groups.keys())

	print "predictions"
	simples = 0
	factors = 0
	medians = 0
	counts = np.zeros(len(test))
	preds = np.zeros(len(test))

	r = 0
	for c, p, d in test[["client_key", "product_key", "depot_key"]].itertuples(False):
		if (c, p) in pair_keys:
			counts[r] = pair_counts[c, p]
			simples += 1
			preds[r] = pair_means[c, p]
			continue
		counts[r] = 0

		if not np.isnan(factor_preds[r]):
			factors += 1
			preds[r] = factor_preds[r]
		else:
			medians += 1
			preds[r] = default
		r += 1
	print "exp()"
	preds = np.exp(preds) - 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (simples, factors, medians)
	return preds, counts


