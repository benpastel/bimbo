import numpy as np, pandas as pd
import pickle, os

def densify(*arrays):
	if len(arrays) > 1: print "\tstacking args..."
	x = np.hstack(arrays)

	print "\tdensifying %d values..." % len(x)
	uniques, indices = np.unique(x, return_inverse=True)
	print "\tmapped each value to a unique value in range(%d)" % len(uniques)

	if len(arrays) == 1: return indices

	out = []
	last_idx = 0
	for a in arrays:
		out.append(indices[last_idx:last_idx + len(a)])
		last_idx += len(a)
	return out

def assert_ndarray(x):
	if not isinstance(x, np.ndarray): raise ValueError("expected ndarray, found " + str(type(x)))

def counts_and_avgs(groups, values, max_group=None):
	assert_ndarray(groups)
	assert_ndarray(values)
	if np.any(np.isnan(values)): raise ValueError("can't handle input NaNs in averaging")
	counts = np.bincount(groups)
	sums = np.bincount(groups, values)
	avgs = sums / counts
	avgs[counts == 0] = np.nan
	if max_group:
		# pad the rest of the values with NaN
		out = np.full(max_group + 1, np.nan)
		out[:len(avgs)] = avgs
		return counts, out
	return counts, avgs

def load_data():
	train_cols = (
		"week",
		"depot_id",
		"client_id",
		"product_id",
		"sales"
	)
	train_dtypes = {
		"week": np.int8,
		"depot_id": np.int32,
		"client_id": np.int32,
		"product_id": np.int32,
		"sales": np.int32
	}
	test_cols = (
		"id",
		"depot_id",
		"client_id",
		"product_id"
	)
	test_dtypes = {
		"id": np.int32,
		"depot_id": np.int32,
		"client_id": np.int32,
		"product_id": np.int32
	}
	indices = ["client_id", "depot_id", "product_id"]
	train_weeks = range(3, 9)

	if not os.path.isfile("pickle/train.pickle"):
		print "loading training data from csv..."
		weekly_data = {}
		for week in range(3, 10):
			with open("split/train_%d.csv" % week, 'r') as f:
				data = pd.read_csv(f, names=train_cols, dtype=train_dtypes, engine='c')
				weekly_data[week] = data
				print "week %d: %d lines" % (week, len(data))
		dev = weekly_data[9]
		train = pd.concat([weekly_data[w] for w in range(3, 9)])

		print "loading test data from csv..."
		test = pd.read_csv("data/slim_test.csv", names = test_cols, dtype = test_dtypes, engine='c')

		print "loading client data from csv..."
		clients = pd.read_csv("data/clients.csv", names = ("client_id", "client_name"), index_col = ("client_id"))

		print "loading product data from csv..."
		products = pd.read_csv("data/products.csv", names = ("product_id", "product_name"), index_col = ("product_id"))

		print "adding log columns"
		train["log_sales"] = np.log(train["sales"] + 1)
		dev["log_sales"] = np.log(dev["sales"] + 1)

		print "mapping client_id, depot_id, product_id into dense keys"
		print "\tmapping..."
		all_depots = list(set(train.depot_id.unique()).union(dev.depot_id.unique()).union(test.depot_id.unique()))
		to_dense_client = {client_id : k for k, client_id in enumerate(clients.index.values)}
		to_dense_product = {product_id : k for k, product_id in enumerate(products.index.values)}
		to_dense_depot = {depot_id : k for k, depot_id in enumerate(all_depots)}
		def densify(frame):
			print "\t\tclients..." 
			client_keys = np.zeros(len(frame), dtype = np.int32)
			client_vals = frame.client_id.values
			for i in range(len(frame)): 
				client_keys[i] = to_dense_client[client_vals[i]]
			frame["client_key"] = client_keys
			print "\t\tproducts..." 
			product_keys = np.zeros(len(frame), dtype = np.int32)
			product_vals = frame.product_id.values
			for i in range(len(frame)):
				product_keys[i] = to_dense_product[product_vals[i]]
			frame["product_key"] = product_keys
			print "\t\tdepots..."
			depot_keys = np.zeros(len(frame), dtype = np.int32)
			depot_vals = frame.depot_id.values
			for i in range(len(frame)):	
				depot_keys[i] = to_dense_depot[depot_vals[i]]
			frame["depot_key"] = depot_keys
		print "\ttrain..."; densify(train)
		print "\tdev..."; densify(dev)
		print "\ttest..."; densify(test)

		print "saving data pickles..."
		train.to_pickle("pickle/train.pickle")
		dev.to_pickle("pickle/dev.pickle")
		test.to_pickle("pickle/test.pickle")
		clients.to_pickle("pickle/clients.pickle")
		products.to_pickle("pickle/products.pickle")

	else:
		print "loading data pickles..."
		train = pd.read_pickle("pickle/train.pickle")
		dev = pd.read_pickle("pickle/dev.pickle")
		test = pd.read_pickle("pickle/test.pickle")
		clients = pd.read_pickle("pickle/clients.pickle")
		products = pd.read_pickle("pickle/products.pickle")

	# if dev_sample:
	# 	dev = dev.sample(n = dev_sample)
	
	print "using %d train, %d dev, %d test lines, with %d clients, %d products" % (
		len(train), len(dev), len(test), len(clients), len(products))
	return train, dev, test, clients, products

def load_no_name_clients():
	lines = pd.read_csv("data/no_name_clients.csv", usecols = [0])
	return set(lines["0"])

def RMSLE(preds, actuals):
	diffs = np.log(preds + 1) - np.log(actuals + 1)
	return np.sqrt( np.average(diffs ** 2) )

if __name__ == '__main__':
	train, dev, test, clients, products = load_data()

