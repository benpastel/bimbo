import numpy as np
import pandas as pd
import os, pickle

from data import counts_and_avgs

def load_product_factors(train, test, data_name):
	path = "pickle/%s_product_factors.pickle" % data_name
	if os.path.isfile(path):
		print "loading %s..." % path
		with open(path, 'r') as f:
			return pickle.load(f)

	print "finding log mean for each client"
	client_avgs = train.groupby("client_id").log_avg.mean()

	print "finding avg price factor for each product relative to the client avg..."
	print "\t client_avg_mask"
	client_avg_mask = client_avgs[train.client_id]

	print "\t price factors mask"
	factors_mask = train.net_units_sold.values / client_avg_mask
	factors_mask[client_avg_mask == 0] = 0

	print "\t grouping by product"
	product_factor_counts, product_factor_avgs = counts_and_avgs(train.product_id, factors_mask)

	print "\t dumping to file"
	with open(path, 'w') as f:
		pickle.dump((product_factor_avgs, client_avgs), f)

	return product_factor_avgs, client_avgs
