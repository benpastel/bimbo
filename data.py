import numpy as np, pandas as pd
import pickle, os

def densify(*arrays):
	x = np.hstack(arrays)

	print "\tdensifying %d values..." % len(x)
	uniques, indices = np.unique(x, return_inverse=True)
	print "\tmapped to range(%d)" % len(uniques)

	if len(arrays) == 1: return indices

	out = []
	last_idx = 0
	for a in arrays:
		out.append(indices[last_idx:last_idx + len(a)].astype(np.int32))
		last_idx += len(a)
	return out

def assert_ndarray(x):
	if not isinstance(x, np.ndarray): raise ValueError("expected ndarray, found " + str(type(x)))

def counts_and_avgs(groups, values, max_group=None):
	if isinstance(groups, pd.Series): groups = groups.values
	if isinstance(values, pd.Series): values = values.values
	assert_ndarray(groups); assert_ndarray(values)
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

# def maxmax(*arrays):
# 	best = np.max(arrays[0])
# 	for i in range(1, len(arrays)):
# 		best = max(best, arrays[i])
# 	return best

def load_data():
	train_cols = (
		"week",
		"depot_id",
		"channel_id", 
		"route_id",
		"client_id",
		"product_id",
		"raw_sales_units",
		"raw_sales_pesos",
		"return_units",
		"return_pesos",
		"net_sales"
	)
	train_dtypes = {
		"week": np.int8,
		"depot_id": np.int32,
		"channel_id": np.int32,
		"route_id": np.int32,
		"client_id": np.int32,
		"product_id": np.int32,
		"raw_sales_units": np.int32,
		"raw_sales_pesos": np.float32,
		"return_units": np.int32,
		"return_pesos": np.float32,
		"net_sales_units": np.int32
	}
	test_cols = (
		"id",
		"depot_id",
		"channel_id", 
		"route_id",
		"client_id",
		"product_id"
	)
	test_dtypes = {
		"id": np.int32,
		"depot_id": np.int32,
		"channel_id": np.int32,
		"route_id": np.int32,
		"client_id": np.int32,
		"product_id": np.int32
	}

	if not os.path.isfile("pickle/train.pickle"):
		print "loading training data from csv..."
		with open("data/train.csv", 'r') as f:
			train = pd.read_csv(f, names=train_cols, dtype=train_dtypes, engine='c')

		print "loading test data from csv..."
		test = pd.read_csv("data/test.csv", names = test_cols, dtype = test_dtypes, engine='c')

		print "loading client data from csv..."
		clients = pd.read_csv("data/clients.csv", names = ("client_id", "client_name"))

		print "loading product data from csv..."
		products = pd.read_csv("data/products.csv", names = ("product_id", "product_name"))

		print "converting everything to log units. remember to convert back before final prediction :-)"
		for col in (
			"raw_sales_units",
			"raw_sales_pesos",
			"return_units",
			"return_pesos",
			"net_sales"):
			train[col] = np.log(train[col] + 1).astype(np.float32)

		for name in (
			"depot",
			"channel", 
			"route"):
			src = name + "_id"
			dst = name + "_key"
			print "converting %s to dense %s" % (src, dst)
			train[dst], test[dst] = densify(train[src], test[src])
			train = train.drop(src, 1)
			test = test.drop(src, 1)

		for name, table in [
			("product", products),
			("client", clients)]:
			src = name + "_id"
			dst = name + "_key"
			print "converting %s to dense %s in 2 passes to remove unseen" % (src, dst)
			train[dst], test[dst], table[dst] = densify(train[src], test[src], table[src])
			is_seen = np.zeros(np.max(table[dst].values) + 1, dtype = np.bool)
			is_seen[train[dst].values] = True
			is_seen[test[dst].values] = True
			seen_keys = np.where(is_seen)[0]
			print "\tseen %d/%d" % (len(seen_keys), len(table))
			table_by_key = table.iloc[ table[dst].values ]
			table = table_by_key.iloc[ seen_keys ].copy()
			print "\tsecond pass"
			train[dst], test[dst], table[dst] = densify(train[dst], test[dst], table[dst])
			train = train.drop(src, 1)
			test = test.drop(src, 1)
			table = table.drop(src, 1)

		print "train columns:"
		print train.dtypes
		print "test columns:"
		print test.dtypes

		print "saving pickles..."
		train.to_pickle("pickle/train.pickle")
		test.to_pickle("pickle/test.pickle")
		clients.to_pickle("pickle/clients.pickle")
		products.to_pickle("pickle/products.pickle")
	else:
		print "loading data pickles..."
		train = pd.read_pickle("pickle/train.pickle")
		test = pd.read_pickle("pickle/test.pickle")
		clients = pd.read_pickle("pickle/clients.pickle")
		products = pd.read_pickle("pickle/products.pickle")

	print "splitting train/dev..."
	dev = train[train.week == 9]
	train = train[train.week < 9]
	
	print "using %d train, %d dev, %d test lines, with %d clients, %d products" % (
		len(train), len(dev), len(test), len(clients), len(products))
	return train, dev, test, clients, products

def load_no_name_clients():
	lines = pd.read_csv("data/no_name_clients.csv", usecols = [0])
	return set(lines["0"])

def RMSE(preds, actuals):
	""" assumes we are already in log space """
	return np.sqrt( np.average((preds - actuals)**2) )

if __name__ == '__main__':
	train, dev, test, clients, products = load_data()

