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

	print "converting to indexless version and hashing (client, depot)"
	no_index = train.reset_index()
	hashes = no_index.client_id.astype(np.int64) * 30000 + no_index.depot_id

	print "finding log mean for (client, depot) (hashed version)"
	_, baseline_avgs = counts_and_avgs(hashes, no_index.log_sales)

	print "broadcasting those averages"
	broadcast = baseline_avgs[hashes]

	print "finding factors"
	factors = no_index.log_sales / broadcast
	factors[broadcast == 0] = np.NaN

	print "averaging by product"
	_, product_factors = counts_and_avgs(no_index.product_id, factors)

	print "\t dumping to file"
	with open(path, 'w') as f:
		pickle.dump((factor_avgs, baseline_avgs), f)

	return factor_avgs, baseline_avgs
