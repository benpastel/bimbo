import numpy as np, pandas as pd
import pickle, os

def counts_and_avgs(groups, values):
	counts = np.bincount(groups)
	sums = np.bincount(groups, values)
	avgs = sums / counts
	avgs[counts == 0] = 0
	return counts, avgs

cached_logs = {x : np.log(x) for x in range(1, 5002)}
def log(x):
	return cached_logs[x]

def load_data(dev_sample=None):
	data_pickle = "pickle/data.pickle"

	train_cols = (
		"week",
		"client_id",
		"product_id",
		"net_units_sold"
	)
	train_dtypes = {
		"week": np.int8,
		"client_id": np.int32,
		"product_id": np.int32,
		"net_units_sold": np.int32
	}
	test_cols = (
		"id",
		"client_id",
		"product_id"
	)
	test_dtypes = {
		"id": np.int32,
		"client_id": np.int32,
		"product_id": np.int32
	}
	train_weeks = range(3, 9)

	if not os.path.isfile(data_pickle):
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
		clients = pd.read_csv("data/clients.csv", names = ("client_id", "client_name"))

		print "loading product data from csv..."
		products = pd.read_csv("data/products.csv", names = ("product_id", "product_name"))

		print "remapping the client_ids as dense keys"
		print "\t finding all sparse ids"
		sparse_clients = set(train.client_id).union(set(dev.client_id)).union(set(test.client_id))
		print "\t mapping to range(%d)" % len(sparse_clients)
		sparse_to_dense = {}
		for i, sparse in enumerate(sparse_clients):
			sparse_to_dense[sparse] = i
		train["client_key"] = np.zeros(len(train), dtype = np.int32)
		dev["client_key"] = np.zeros(len(dev), dtype = np.int32)
		test["client_key"] = np.zeros(len(test), dtype = np.int32)
		clients["client_key"] = np.zeros(len(clients), dtype = np.int32)
		print "\t adding columns to dataframes"
		for i, sparse in enumerate(train.client_id): train["client_key"].values[i] = sparse_to_dense[sparse]
		for i, sparse in enumerate(dev.client_id): 	dev["client_key"].values[i] = sparse_to_dense[sparse]
		for i, sparse in enumerate(test.client_id):	test["client_key"].values[i] = sparse_to_dense[sparse]
		for i, sparse in enumerate(clients.client_id): clients["client_key"].values[i] = sparse_to_dense.get(sparse, None)

		print "adding dense (client, product) pair keys"
		print "\t finding all sparse ids"
		sparse_pair = lambda frame: 50000 * frame.client_key.astype(np.int64) + frame.product_id
		pairs = set(sparse_pair(train)).union(set(sparse_pair(dev))).union(set(sparse_pair(test)))
		print "\t mapping to range(%d)" % len(pairs)
		sparse_to_dense = {}
		for i, sparse in enumerate(pairs):
			sparse_to_dense[sparse] = i
		train["pair_key"] = np.zeros(len(train), dtype = np.int64)
		dev["pair_key"] = np.zeros(len(dev), dtype = np.int64)
		test["pair_key"] = np.zeros(len(test), dtype = np.int64)
		print "\t adding columns to dataframes"
		for i, sparse in enumerate(sparse_pair(train)): train.pair_key.values[i] = sparse_to_dense[sparse]
		for i, sparse in enumerate(sparse_pair(dev)): 	dev.pair_key.values[i] = sparse_to_dense[sparse]
		for i, sparse in enumerate(sparse_pair(test)):	test.pair_key.values[i] = sparse_to_dense[sparse]

		print "saving data pickle..."
		with open(data_pickle, 'w') as f:
			pickle.dump((train, dev, test, clients, products), f)
	else:
		print "loading data pickle..."
		with open(data_pickle, 'w') as f:
			train, dev, test, clients, products = pickle.load(f)

	if dev_sample:
		dev = dev.sample(n = dev_sample)
	
	print "using %d train, %d dev, %d test lines" % (len(train), len(dev), len(test))
	return train, dev, test, clients, products

def RMSLE(preds, actuals):
	diffs = np.log(preds + 1) - np.log(actuals + 1)
	return np.sqrt( np.average(diffs ** 2) )

if __name__ == '__main__':
	train, dev, test, clients, products = load_data()

