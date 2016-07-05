import numpy as np
import pandas as pd

from product_factors import load_product_factors
from data import counts_and_avgs, log, load_data

GLOBAL_MEDIAN = 3 # TODO use more precision

def int_avg(trains, devs):
	return predict_avg(trains, devs, 
		lambda sales: sales, 
		lambda total, count: total / count)

def simple_avg(trains, devs):
	return predict_avg(trains, devs, 
		lambda sales: sales, 
		lambda total, count: round(float(total) / count, 2))

def log_avg(trains, devs):
	return predict_avg(trains, devs,
		lambda sales: log(sales + 1),
		lambda total, count: 
			round(np.exp(total / count) - 1, 2))

def log_avg_product_factors(trains, devs):
	return predict_avg_with_price_factors(trains, devs,
		lambda sales: log(sales + 1),
		lambda total, count: 
		round(np.exp(total / count) - 1, 2))

def predict_avg(trains, devs, encode_fn, decode_fn):
	sales_sum = {}
	sales_count = {}

	print "scanning dev set for (client, product) pairs"
	for key in devs["key"]:
		sales_sum[key] = 0
		sales_count[key] = 0
	print "%d pairs / %d lines" % (len(sales_count), len(devs))

	print "building avgs from train set"
	used = 0
	trimmed = trains[["key", "net_units_sold"]]

	for key, sales in trimmed.itertuples(False):
		if key in sales_count:
			sales_sum[key] += encode_fn(sales)
			sales_count[key] += 1
			used += 1
	print "used %d / %d lines" % (used, len(trains))

	print "making predictions"
	hits = 0
	misses = 0
	preds = np.zeros(len(devs))
	counts = np.zeros(len(devs))
	r = 0
	for key in devs["key"]:
		counts[r] = sales_count[key]
		if sales_count[key] > 0:
			hits += 1
			preds[r] = decode_fn(sales_sum[key], sales_count[key])
		else:
			misses += 1
			preds[r] = median
		r += 1
	print "hit %d pairs, fell back to median for %d" % (hits, misses)
	return preds, counts

# TODO: try specifying minimum sample sizes for using this approach
# TODO: where do I need to check for inf???
def predict_avg_with_price_factors(train, test, data_name):
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
	for pair, c, p, in test[["pair_key", "client_key", "product_id"]].itertuples(False):
		counts[r] = sales_count[key]
		if sales_avg[pair] > 0:
			pair_avgs += 1
			preds[r] = sales_avg[pair]
		elif product_factors[p] > 0 and client_avgs[c] > 0:
			factors += 1
			preds[r] = client_avgs[c] * product_factors[p]
		else:
			medians += 1
			preds[r] = median
		r += 1
	print "used: %d simple avg, %d product_factor * client_avg, %d median" % (pair_avgs, factors, medians)
	return preds, counts


if __name__ == '__main__':
	print "DEBUG ONLY"
	train, dev, test = load_data()
	train = pd.concat((train, dev))
	predict_avg_with_price_factors(train, test, "for_test")


