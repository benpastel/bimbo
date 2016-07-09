import numpy as np
import pandas as pd

from product_factors import load_product_factors
from data import counts_and_avgs, log, load_data, load_no_name_clients

GLOBAL_MEDIAN = 3

def reference(train, test, clients, products, data_name):
	return predict_reference(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def current(train, test, clients, products, data_name):
	return predict_current(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def predict_current(train, test, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	# TODO compare vs. median
	print "finding log avgs by (client, depot, product)"
	print "\tcounts"
	triple_counts = train.groupby(level=["client_id", "depot_id", "product_id"]).log_avg.count()
	print "\tmeans"
	triple_means = train.groupby(level=["client_id", "depot_id", "product_id"]).log_avg.mean()

	print "making predictions"
	triple_avgs = 0
	factors = 0
	medians = 0
	preds = np.zeros(len(test))
	counts = np.zeros(len(test))

	r = 0
	for c, d, p in test[["client_id", "depot_id", "product_id"]].itertuples(False):
		if (c, d, p) in triple_means:
			counts[r] = triple_counts[c, d, p]
			triple_avgs += 1
			preds[r] = triple_means[c, d, p]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			counts[r] = 0
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			counts[r] = 0
			medians += 1
			preds[r] = GLOBAL_MEDIAN
		r += 1
	print "exp()"
	preds = np.exp(preds) - 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (triple_avgs, factors, medians)
	return preds, counts

def predict_reference(train, test, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	print "finding log avgs by (client, product)"
	print "\tcounts"
	pair_counts = train.groupby(level=["client_id", "product_id"]).log_avg.count()
	print "\tmeans"
	pair_means = train.groupby(level=["client_id", "product_id"]).log_avg.mean()
	sales_count, sales_avg = counts_and_avgs(train.pair_key.values, train.net_units_sold.values)

	print "making predictions"
	pair_avgs = 0
	factors = 0
	medians = 0
	preds = np.zeros(len(test))
	counts = np.zeros(len(test))

	r = 0
	for c, p in test[["client_id", "product_id"]].itertuples(False):
		if (c, p) in pair_means:
			counts[r] = pair_counts[c, p]
			triple_avgs += 1
			preds[r] = pair_means[c, p]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			counts[r] = 0
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			counts[r] = 0
			medians += 1
			preds[r] = GLOBAL_MEDIAN
		r += 1
	print "exp()"
	preds = np.exp(preds) - 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (pair_avgs, factors, medians)
	return preds, counts


