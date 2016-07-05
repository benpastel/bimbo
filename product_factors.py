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

	print "finding mean for each client"
	client_counts, client_avgs = counts_and_avgs(train.client_key, train.net_units_sold)

	print "finding avg price factor for each product relative to the client avg..."
	print "\t client_avg_mask"
	client_avg_mask = client_avgs[train.client_id.values]

	print "\t price factors"
	price_factors = train.net_units_sold.values / client_avg_mask # TODO: deal with these inf values first

	print "\t dumping to file"
	with open(path, 'w') as f:
		pickle.dump((price_factors, client_avgs), f)

	return price_factors, client_avgs
