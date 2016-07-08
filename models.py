import numpy as np
import pandas as pd

from product_factors import load_product_factors
from data import counts_and_avgs, log, load_data, load_no_name_clients

GLOBAL_MEDIAN = 3

def reference(raw_train, test, clients, products, data_name):
	return predict_reference(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def current(raw_train, test, clients, products, data_name):
	return predict_current(train, test, clients, products, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

def predict_reference(train, test, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	# TODO compare vs. median
	print "finding log avgs by (client, depot, product)"
	print "\tcounts"
	triple_counts = train.groupby(["client_id", "depot_id", "product_id"]).log_avg.count()
	print "\tmeans"
	triple_means = train.groupby(["client_id", "depot_id", "product_id"]).log_avg.mean()

	print "making predictions"
	triple_avgs = 0
	factors = 0
	medians = 0
	preds = np.zeros(len(test))
	counts = np.zeros(len(test))

	r = 0
	for c, d, p in test[["client_id", "depot_id", "product_id"]].itertuples(False):
		if (c, d, p) in triple_means:
			triple_avgs += 1
			preds[r] = triple_means[c, d, p]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			medians += 1
			preds[r] = GLOBAL_MEDIAN
		r += 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (pair_avgs, factors, medians)

	return preds, counts

def predict_current(train, test, clients, products, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	print "finding averages for each pair"
	sales_count, sales_avg = counts_and_avgs(train.pair_key.values, train.net_units_sold.values)

	no_name_clients = load_no_name_clients()

	print "making predictions"
	pair_avgs = 0
	factors = 0
	medians = 0
	preds = np.zeros(len(test))
	counts = np.zeros(len(test))
	no_names = 0

	r = 0
	for pair, c, p, client_id in test[["pair_key", "client_key", "product_id", "client_id"]].itertuples(False):
		counts[r] = sales_count[pair]

		if c in no_name_clients:
			no_names += 1
			preds[r] = GLOBAL_MEDIAN
		elif sales_avg[pair] > 0:
			pair_avgs += 1
			preds[r] = sales_avg[pair]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			medians += 1
			preds[r] = GLOBAL_MEDIAN
		r += 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median, %d no_names" % (
		pair_avgs, factors, medians, no_names)
	return preds, counts


