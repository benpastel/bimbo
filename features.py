import numpy as np
import pandas as pd

from product_factors import *
from data import counts_and_avgs, load_data, densify

def feature_defs(clients, products):
	return (client_features(clients) + product_features(products) + pair_key_features() + single_key_features() + [
		("last_nonzero_logsale", last_nonzero_logsale),
		("weeks_since_last_sale", weeks_since_last_sale),
		("logsale_last_week", logsale_last_week),
		("clientname_product avgs", lambda train, test: client_name_features(train, test, clients)),
		("product factors vs client", lambda train, test: product_factor_vs_client_features(train, test)),
		("client factors vs product", lambda train, test: client_factor_vs_product_features(train, test))
	])

def client_features(clients):
	return []

def product_features(products):
	return []

def last_nonzero_logsale(train, test):
	key_fn = product_client_hash
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	by_key = np.zeros(max_key + 1, dtype = np.int32)

	test_week = extract_week(test)
	for week in range(3, test_week):
		# overwrite values with the most recent week
		ok = (train.week.values == week) & (train.net_sales.values > 0)
		k = train_keys[ok]
		by_key[k] = train.net_sales.values[ok]
	return by_key[test_keys]

# TODO handle 10 / 11 test weeks differently
def extract_week(test):
	# return test.week.values[0]
	if "week" in test.columns and test.week.values[0] <= 10:
		out = test.week.values[0]
	else:
		out = 10
	print "\t(pretending all of test is week %d)" % out
	return out

def weeks_since_last_sale(train, test):
	key_fn = product_client_hash
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	# start all the keys at 10
	since = np.zeros(max_key + 1, dtype = np.int32) + 10

	test_week = extract_week(test)

	for week in range(3, test_week):
		ago = test_week - week
		k = train_keys[train.week.values == week]
		since[k] = ago
	out = since[test_keys]
	for ago in range(11):
		print "\t\tweeks since last sale: %d: %d" % (ago, np.count_nonzero(out == ago))
	return out

def single_key_features():
	names = [
		"depot",
		"channel", 
		"route",
		"client",
		"product"]
	def col_avg(col):
		def feature(train, test):
			max_key = max(train[col].max(), test[col].max())
			print "\tmax key:", max_key
			print "\tpooling log means"
			_, means = counts_and_avgs(train[col], train.net_sales.values, max_group = max_key)
			return means[test[col]]
		return feature

	builders = []
	for name in names:
		col = name + "_key"
		feature_name = name + "_avg"
		feature = col_avg(col)
		builders.append((feature_name, feature))
	return builders

def pair_key_features():
	names = [
		"depot",
		"channel", 
		"route",
		"client",
		"product"]
	builders = []
	def pair_avg(col1, col2):
		def feature(train, test):
			col2_max = max(train[col2].max(), test[col2].max())
			print "\tkey factor:", (col2_max + 1)
			return avg_by_key(train, test, 
				lambda frame: frame[col1].astype(np.int64) * (col2_max + 1) + frame[col2])
		return feature

	for i in range(len(names)):
		for j in range(i+1, len(names)):
			name1, name2 = names[i], names[j]
			feature_name = "%s_%s_avg" % (name1, name2)
			col1, col2 = name1 + "_key", name2 + "_key"
			builders.append((feature_name, pair_avg(col1, col2)))
	return builders

def logsale_last_week(train, test):
	key_fn = product_client_hash
	week = extract_week(test) - 1
	print "\tfinding trains in week %d" % week
	last_train = train[train.week == week]
	print "\t%d/%d trains are last week" % (len(last_train), len(train))
	print "\tbuilding keys"
	train_keys, test_keys = densify(key_fn(last_train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	print "\tlooking up vals"
	vals = np.zeros(max_key + 1, dtype = np.int32)
	vals[train_keys] = last_train.net_sales.values
	out = vals[test_keys]
	print "\t%d/%d nonzero vals" % (np.count_nonzero(out), len(test))
	return out

def avg_by_key(train, test, key_fn):
	print "\tbuilding keys"
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))
	print "\tmax key:", max_key

	print "\tpooling log means"
	_, means = counts_and_avgs(train_keys, train.net_sales.values, max_group = max_key)

	return means[test_keys]

def client_name_features(train, test, clients):	
	print "\t\thashing client names"
	client_hashes = np.zeros(np.max(clients.client_key) + 1, dtype=np.int64)
	for r, name in enumerate(clients.client_name):
		client_key = clients.client_key[r]
		client_hashes[client_key] = hash(name)

	def key(frame):
		return (client_hashes[frame.client_key] * 3000
			+ frame.product_key.values)
	train_keys, test_keys = densify(key(train), key(test))

	_, means = counts_and_avgs(train_keys, train.net_sales.values)
	return means[test_keys]

def product_client_hash(frame):
	return frame.client_key.values.astype(np.int64) * 3000 + frame.product_key.values

def product_client_depot_hash(frame):
	return (frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

def product_depot_hash(frame):
	return frame.product_key.values * 600 + frame.depot_key.values



