import numpy as np, pandas as pd
import pickle, os

def counts_and_avgs(groups, values):
	counts = np.bincount(groups)
	sums = np.bincount(groups, values)
	avgs = sums / counts
	avgs[counts == 0] = np.NaN
	return counts, avgs

cached_logs = {x : np.log(x) for x in range(1, 5002)}
def log(x):
	return cached_logs[x]

def load_data(dev_sample=None):
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
				data = pd.read_csv(f, names=train_cols, dtype=train_dtypes, index_col = indices, engine='c')
				weekly_data[week] = data
				print "week %d: %d lines" % (week, len(data))
		dev = weekly_data[9]
		train = pd.concat([weekly_data[w] for w in range(3, 9)])

		print "loading test data from csv..."
		test = pd.read_csv("data/slim_test.csv", names = test_cols, dtype = test_dtypes, index_col = indices, engine='c')

		print "loading client data from csv..."
		clients = pd.read_csv("data/clients.csv", names = ("client_id", "client_name"), index_col = ("client_id"))

		print "loading product data from csv..."
		products = pd.read_csv("data/products.csv", names = ("product_id", "product_name"), index_col = ("product_id"))

		print "adding log columns"
		train["log_sales"] = np.log(train["sales"] + 1)
		dev["log_sales"] = np.log(dev["sales"] + 1)

		print "mapping client_id into dense client_row"
		print "\tmapping..."
		sparse_to_dense = {client_id : row for row, client_id in enumerate(clients.index.values)}
		def add_client_row(frame):
			client_rows = np.zeros(len(frame), dtype = np.int32)
			ids = frame.index.get_level_values("client_id")
			for i in range(len(frame)): 
				client_rows[i] = sparse_to_dense[ids[i]]
			frame["client_rows"] = client_rows
		print "\ttrain..."; add_client_row(train)
		print "\tdev..."; add_client_row(dev)
		print "\ttest..."; add_client_row(test)

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

	if dev_sample:
		dev = dev.sample(n = dev_sample)
	
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

