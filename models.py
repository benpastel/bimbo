import numpy as np
import pandas as pd

from product_factors import load_product_factors
from data import counts_and_avgs, log, load_data

GLOBAL_MEDIAN = 3 # TODO use more precision

def avg_pair_avg_product_factors(train, test, data_name):
	return predict_avg_with_product_factors(train, test, data_name, GLOBAL_MEDIAN)

def logavg_pair_logavg_product_factors(raw_train, test, data_name):

	print "copying train..."
	train = raw_train.copy()

	print "mapping into log space"
	train.net_units_sold = np.log(raw_train.net_units_sold + 1)
	raw_pred, counts = predict_avg_with_product_factors(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

	print "mapping back out of log space"
	pred = np.exp(raw_pred) - 1
	return pred, counts

def current(raw_train, test, data_name):
	print "copying train..."
	train = raw_train.copy()

	print "mapping into log space"
	train.net_units_sold = np.log(raw_train.net_units_sold + 1)
	raw_pred, counts = predict_current(train, test, data_name + "_logs", np.log(GLOBAL_MEDIAN + 1))

	print "mapping back out of log space"
	pred = np.exp(raw_pred) - 1
	return pred, counts


# TODO: try specifying minimum sample sizes for using this approach
def predict_avg_with_product_factors(train, test, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	print "finding averages for each pair"
	sales_count, sales_avg = counts_and_avgs(train.pair_key.values, train.net_units_sold.values)

	print "making predictions"
	pair_avgs = 0
	factors = 0
	medians = 0
	preds = np.zeros(len(test))
	counts = np.zeros(len(test))

	r = 0
	for pair, c, p, client_id in test[["pair_key", "client_key", "product_id", "client_id"]].itertuples(False):
		counts[r] = sales_count[pair]

		if sales_avg[pair] > 0:
			pair_avgs += 1
			preds[r] = sales_avg[pair]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			medians += 1
			preds[r] = GLOBAL_MEDIAN
		r += 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (pair_avgs, factors, medians)


	return preds, counts

def predict_current(train, test, data_name, default):
	product_factors, client_avgs = load_product_factors(train, test, data_name)

	print "finding averages for each pair"
	sales_count, sales_avg = counts_and_avgs(train.pair_key.values, train.net_units_sold.values)

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
		if client_id == 0 or p == 0: # 'SIN NOMBRE' or 'NO IDENTIFICADO'
			# TODO: actually there are bunch of other SIN NOMBREs.  need to actually import the client table.
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


