import numpy as np
import pandas as pd

from product_factors import *
from data import *

def feature_defs(clients, products):
	# return (
	# 	all_pairwise_factor_features(KEY_COLUMNS)
	# 	+ pair_key_features() 
	# 	+ single_key_features() + [
	# 	("last_nonzero_sale", last_nonzero_logsale),
	# 	("sale_last_week", logsale_last_week),
	# ])
	return single_key_features(clients, products)

def client_features(clients):
	return [("client_includes_no_name", client_includes_no_name(clients))]

def product_features(products):
	return []

# TODO: parse the client names as strings from the beginning
def client_includes_no_name(clients):
	def gen(train, test):
		print "\tchecking which clients have 'SIN NOMBRE'"
		find = np.core.defchararray.find
		names = np.array(clients.client_name.values, dtype=np.str)
		no_name = (-1 != find(names, 'SIN NOMBRE'))
		print "\t%d/%d clients have 'SIN NOMBRE'" % (np.count_nonzero(no_name), len(no_name))
		print "\tbroadcasting to test"
		by_key = np.zeros(np.max(clients.client_key) + 1, dtype=np.bool)
		by_key[clients.client_key.values] = no_name
		test_no_name = by_key[test.client_key]
		print "\t%d/%d test rows have 'SIN NOMBRE'" % (np.count_nonzero(test_no_name), len(test))
		return test_no_name
	return gen

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

def as_fn(col):
	return lambda frame: frame[col]

def single_key_features(clients, products):
	keys = [
		("clientname", clientname_hash_fn(clients)),
		("depot", as_fn("depot_key")),
		("channel", as_fn("channel_key")),
		("route", as_fn("route_key")),
		("client", as_fn("client_key")),
		("product", as_fn("product_key"))
	]
	def feature(key_fn):
		def f(train, test):
			train_keys, test_keys = key_fn(train), key_fn(test)
			max_key = max(np.max(train_keys), np.max(test_keys))
			print "\tmax key:", max_key
			print "\tpooling log means"
			_, means = counts_and_avgs(train_keys, train.net_sales.values, max_group = max_key)
			return means[test_keys]
		return f
	builders = []
	for (name, fn) in keys:
		builders.append((name + "_avg", feature(fn)))
	return builders

def pair_key_features():
	names = [
		"depot",
		"channel", 
		"route",
		"client",
		"product"
	]
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

# cache of all clientname hashes indexed by client_key
CLIENTNAME_HASHES = None
def clientname_hash_fn(clients):
	global CLIENTNAME_HASHES
	""" returns a function (frame => clientname_hashes) """
	if not CLIENTNAME_HASHES:
		print "\thashing client names"
		hashes = densify(clients.client_name.values)
		keys = clients.client_key.values

		# index by client key
		CLIENTNAME_HASHES = np.zeros(np.max(keys) + 1, dtype = np.int32)
		for i in range(len(hashes)):
			CLIENTNAME_HASHES[keys[i]] = hashes[i]
	return lambda frame: CLIENTNAME_HASHES[frame.client_key]

def product_client_hash(frame):
	return frame.client_key.values.astype(np.int64) * 3000 + frame.product_key.values

def product_client_depot_hash(frame):
	return (frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

def product_depot_hash(frame):
	return frame.product_key.values * 600 + frame.depot_key.values



