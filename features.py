import numpy as np
import pandas as pd

from product_factors import *
from data import *

from partitioning import partition_feature_defs, as_fn, clientname_hash_fn

def feature_defs(clients, products):
	print "preparing feature definitions"
	return (
		client_features(clients)
		+ pairwise_factor_features(clients, products)
		+ pair_key_features(clients, products) 
		+ single_key_features(clients, products) 
		+ [
			("last_nonzero_sale", last_nonzero_logsale),
			("sale_last_week", logsale_last_week)
		]
		+ partition_feature_defs(clients, products)
		)

def client_features(clients):
	def length(names): 
		return np.char.count(names, '') - 1

	return [
		("clientname_length", clientname_feature(clients, length)),
		]

def product_features(products):
	return []

def pairwise_factor_features(clients, products):
	keys = {
		"clientname": clientname_hash_fn(clients),
		"depot": as_fn("depot_key"),
		"channel": as_fn("channel_key"),
		"route": as_fn("route_key"),
		"client": as_fn("client_key"),
		"product": as_fn("product_key")
	}

	# pairs = [
	# 	('product', 'client'),
	#     ('client', 'product'),
	#     ('product', 'channel'),
	#     ('product', 'depot'),
	#     ('channel', 'route'),
	#     ('route', 'product'),
	#     ('depot', 'route'),
	#     ('depot', 'product'),
	#     ('channel', 'product'),
	#     ('product', 'route'),
	#     ('route', 'channel'),
	#     ('depot', 'channel'),
	#     ('client', 'route'),
	#     ('route', 'depot'),
	#     ('depot', 'clientname'),
	#     ('clientname', 'route'),
	#     ('client', 'depot'),
	#     ('depot', 'client'),
	#     ('route', 'clientname'),
	#     ('clientname', 'client'),
	#     ('product', 'clientname'),
	#     ('clientname', 'depot'),
	#     ('channel', 'depot'),
	#     ('clientname', 'product'),
	# ]
	def feature(key1, key2):
		return lambda train, test: avg_factor_features(train, test, key1, key2)

	features = []
	for i in range(len(keys)):
		for j in range(i+1, len(keys)):
			name1 = keys.keys()[i]
			name2 = keys.keys()[j]
			fn1 = keys[name1]
			fn2 = keys[name2]
			features.append(("%s_vs_%s_factors" % (name1, name2), feature(fn1, fn2)))
	return features

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

def single_key_features(clients, products):
	keys = [
		("clientname", clientname_hash_fn(clients)),
		("depot", as_fn("depot_key")),
		("channel", as_fn("channel_key")), # might be useless
		("route", as_fn("route_key")),
		("client", as_fn("client_key")), # might be useless
		("product", as_fn("product_key"))
	]
	def avg_feature(key_fn):
		def f(train, test):
			train_keys, test_keys = key_fn(train), key_fn(test)
			max_key = max(np.max(train_keys), np.max(test_keys))
			_, means = counts_and_avgs(train_keys, train.net_sales.values, max_group = max_key)
			return means[test_keys]
		return f
	def count_feature(key_fn):
		def f(train, test):
			train_keys, test_keys = key_fn(train), key_fn(test)
			max_key = max(np.max(train_keys), np.max(test_keys))
			counts, _ = counts_and_avgs(train_keys, train.net_sales.values, max_group = max_key)
			return counts[test_keys]
		return f
	builders = []
	for (name, fn) in keys:
		builders.append((name + "_avg", avg_feature(fn)))
		builders.append((name + "_count", count_feature(fn)))
	return builders

def pair_key_features(clients, products):
	keys = {
		"clientname": clientname_hash_fn(clients),
		"depot": as_fn("depot_key"),
		"channel": as_fn("channel_key"),
		"route": as_fn("route_key"),
		"client": as_fn("client_key"),
		"product": as_fn("product_key")
	}
	# pairs = []
	# # add product vs all others
	# for (name, fn) in keys.iteritems():
	# 	if name != "product": 
	# 		pairs.append(("product", name))
	# pairs += [
	# 	("route", "client"),
	# 	("depot", "route"),
	# 	("clientname", "route"),
	# 	("depot", "client"),
	# 	("depot", "channel"),
	# 	("channel", "route"),
	# 	("clientname", "depot"),
	# 	("channel", "client"),
	# 	("clientname", "channel")
	# ]

	def avg_feature(key1, key2):
		def f(train, test):
			train_key1s, test_key1s = key1(train), key1(test)
			train_key2s, test_key2s = key2(train), key2(test)
			key2_max = max(np.max(train_key2s), np.max(test_key2s))
			print "\tkey factor:", (key2_max + 1)
			return avg_by_key(train, test, 
				lambda frame: key1(frame).astype(np.int64) * (key2_max + 1) + key2(frame))
		return f

	def count_feature(key1, key2):
		def f(train, test):
			train_key1s, test_key1s = key1(train), key1(test)
			train_key2s, test_key2s = key2(train), key2(test)
			key2_max = max(np.max(train_key2s), np.max(test_key2s))
			combined_key = lambda frame: key1(frame).astype(np.int64) * (key2_max + 1) + key2(frame)

			train_keys, test_keys = densify(combined_key(train), combined_key(test))
			counts = np.bincount(train_keys)
			return counts[test_keys]
		return f

	builders = []
	for i in range(len(keys)):
		for j in range(i+1, len(keys)):
			name1 = keys.keys()[i]
			name2 = keys.keys()[j]
			fn1 = keys[name1]
			fn2 = keys[name2]
			builders.append(("%s_%s_avg" % (name1, name2), avg_feature(fn1, fn2)))
			builders.append(("%s_%s_count" % (name1, name2), count_feature(fn1, fn2)))

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

def clientname_feature(clients, clientname_fn):
	print "\tmapping over clientnames:", clientname_fn
	vals = clientname_fn(np.array(clients.client_name.values, dtype = np.str))
	vals_by_key = np.zeros(np.max(clients.client_key) + 1, dtype = np.float32)
	keys = clients.client_key
	for i in range(len(vals)):
		vals_by_key[keys[i]] = vals[i]

	return lambda train, test: vals_by_key[test.client_key]

def product_client_hash(frame):
	return frame.client_key.values.astype(np.int64) * 3000 + frame.product_key.values

def product_client_depot_hash(frame):
	return (frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

def product_depot_hash(frame):
	return frame.product_key.values * 600 + frame.depot_key.values



