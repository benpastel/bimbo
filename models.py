import numpy as np
import pandas as pd

def simple_avg(trains, devs, encode=lambda x:x, decode=lambda x:x):
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
	for key, sales in trains[("key", "net_units_sold")]:
		if key in sales_count:
			sales_sum[key] += sales
			sales_count[key] += 1
			used += 1
	print "used %d / %d lines" % (used, len(trains))

	print "making predictions"
	hits = 0
	misses = 0
	preds = np.zeros(len(devs))
	r = 0
	for key in devs["key"]:
		if sales_count[key] > 0:
			hits += 1
			preds[r] = sales_sum[key] / sales_count[key]
		else:
			misses += 1
			preds[r] = median
		r += 1
	print "hit %d pairs, fell back to median for %d" % (hits, misses)
	return preds