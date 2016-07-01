import numpy as np
import pandas as pd

cached_logs = {x : np.log(x) for x in range(1, 5002)}
def log(x):
	return cached_logs[x]

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

def predict_avg(trains, devs, encode_fn, decode_fn):
	median = 3
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