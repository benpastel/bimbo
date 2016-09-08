import numpy as np
import pandas as pd
import random
import datetime

from data import *

def as_fn(col):
	return lambda frame: frame[col]

# cache of all clientname hashes indexed by client_key
CLIENTNAME_HASHES = None
def clientname_hash_fn(clients):
	global CLIENTNAME_HASHES
	""" returns a function (frame => clientname_hashes) """
	if CLIENTNAME_HASHES is None:
		print "\tpreprocess client names"
		names = np.array(clients.client_name.values, dtype = np.str)
		for token in ['EL ', 'LA ', 'LAS ', 'DE ', 'LOS ']:
			names = np.char.replace(names, token, '')
		names = np.char.strip(names)

		print "\thashing client names"
		hashes = densify(names)
		keys = clients.client_key.values

		# index by client key
		CLIENTNAME_HASHES = np.zeros(np.max(keys) + 1, dtype = np.int32)
		for i in range(len(hashes)):
			CLIENTNAME_HASHES[keys[i]] = hashes[i]
	return lambda frame: CLIENTNAME_HASHES[frame.client_key.values]

MAX_PARTITION_LEVELS = 6
def partition_feature_defs(clients, products):
	all_key_fns = {
		"clientname": clientname_hash_fn(clients),
		"depot": as_fn("depot_key"),
		"channel": as_fn("channel_key"),
		"route": as_fn("route_key"),
		"client": as_fn("client_key"),
		"product": as_fn("product_key")
	}

	good_results = [
		('partition_100_channel_product_client_depot_clientname_route', 243),
		('partition_1000_channel_product_depot_route_clientname_client', 181),
		('partition_300_route_depot_product_client_channel_clientname', 121),
		('partition_300_product_client_channel_clientname_route_depot', 85),
		('partition_1000_route_depot_channel_client_product_clientname', 59),
		('partition_300_depot_product_clientname_client_route_channel', 58),
		('partition_100_client_channel_clientname_depot_route_product', 53),
		('partition_1000_channel_product_route_client_depot_clientname', 49),
		('partition_100_client_product_route_depot_clientname_channel', 47),
		('partition_300_route_client_channel_clientname_product_depot', 38),
		('partition_300_client_product_route_depot_clientname_channel', 31),
		('partition_100_channel_depot_client_clientname_product_route', 26),
		('partition_100_channel_depot_route_client_clientname_product', 22),
		('partition_300_client_depot_clientname_channel_route_product', 21),
		('partition_100_route_client_channel_clientname_depot_product', 20),
		('partition_1000_client_channel_product_route_depot_clientname', 20),
		('partition_1000_channel_clientname_client_depot_product_route', 15),
		('partition_1000_channel_depot_clientname_route_client_product', 15),
		('partition_100_client_depot_clientname_channel_product_route', 14),
		('partition_100_clientname_route_product_depot_client_channel', 13),
		('partition_100_product_clientname_client_depot_route_channel', 13),
		('partition_1000_clientname_product_route_channel_client_depot', 10),
		('partition_1000_clientname_depot_channel_client_route_product', 10),
		('partition_300_product_clientname_route_client_depot_channel', 9),
		('partition_300_clientname_product_client_channel_depot_route', 7),
		('partition_300_product_channel_route_clientname_client_depot', 6),
		('partition_300_channel_clientname_product_client_route_depot', 4),
		('partition_1000_clientname_product_route_depot_client_channel', 3),
		('partition_1000_clientname_depot_client_product_route_channel', 2),
	]
	feats = [] 
	for (name, _) in good_results:
		_, size, level_string = name.split('_', 2)
		levels = level_string.split('_')
		feats.append(make_partition_feature(levels, all_key_fns, int(size)))
	return feats

VERBOSE = False
def make_partition_feature(level_order, all_key_fns, max_group_size):
	name = "partition_" + str(max_group_size) + "_" + "_".join(level_order[0:MAX_PARTITION_LEVELS])

	def f(train, test):
		tic = datetime.datetime.now()
		is_train_unfinished = np.ones((len(train), ), dtype=np.bool)
		is_test_unfinished = np.ones((len(test), ), dtype=np.bool)

		final_avgs = np.zeros((len(test), ), dtype=np.float32)

		train_keys = np.zeros((len(train), ), dtype=np.int64)
		test_keys = np.zeros((len(test), ), dtype=np.int64)

		level_count = 0
		levels = []
		while (level_count < MAX_PARTITION_LEVELS
			and np.any(is_train_unfinished) 
			and np.any(is_test_unfinished)):
		
			level = level_order[level_count]
			level_count += 1
			levels.append(level)

			print "\tpartioning on", level

			# extract the latest keys
			# TODO: precompute these!!
			key_fn = all_key_fns[level]
			new_train_keys = is_train_unfinished * key_fn(train)
			new_test_keys = is_test_unfinished * key_fn(test)
		
			# shift previous keys out of the way
			shift = max(np.max(new_train_keys), np.max(new_test_keys)) + 1
			train_keys *= shift
			test_keys *= shift
			train_keys += (new_train_keys + 1)
			test_keys += (new_test_keys + 1)

			if np.any(train_keys < 0) or np.any(test_keys < 0):
				print "dtypes:", train_keys.dtype, test_keys.dtype
				raise Exception("Overflow!")

			unfinished_train_count = np.count_nonzero(is_train_unfinished)

			# group by just the new keys
			all_keys = np.hstack([	train_keys[is_train_unfinished], 
									test_keys[is_test_unfinished]	])
			uniques, indices = np.unique(all_keys, return_inverse=True)

			train_keys[is_train_unfinished] = indices[0:unfinished_train_count]
			test_keys[is_test_unfinished] = indices[unfinished_train_count:]

			counts, avgs = counts_and_avgs(train_keys, train.net_sales.values, len(uniques))

			new_group_unfinished = (counts > max_group_size)
			if VERBOSE:
				print "\t\t%d/%d new groups unfinished (+/- 1)" % (np.count_nonzero(new_group_unfinished), len(new_group_unfinished))

			test_is_nan = (counts[test_keys] == 0)
			if VERBOSE: 
				print "\t\t%d/%d test keys became nan; keeping previous estimate (or 0)" % (np.count_nonzero(test_is_nan), len(test))
			is_test_unfinished &= ~test_is_nan

			final_avgs[is_test_unfinished] = avgs[test_keys][is_test_unfinished]

			is_train_unfinished &= new_group_unfinished[train_keys]
			is_test_unfinished &= new_group_unfinished[test_keys]
			train_keys *= is_train_unfinished
			test_keys *= is_test_unfinished

			if VERBOSE:
				print "\t\t%d/%d trains unfinished" % (np.count_nonzero(is_train_unfinished), len(train))
				print "\t\t%d/%d tests unfinished" % (np.count_nonzero(is_test_unfinished), len(test))

		if VERBOSE:
			print "\t\t%d/%d final_avgs are nonzero" % (np.count_nonzero(final_avgs), len(final_avgs))
			print "elapsed: %s ms" % (datetime.datetime.now() - tic)
		if np.any(np.isnan(final_avgs)): 
			raise ValueError("This feature shouldn't have any NaN values")
		if np.any(final_avgs < 0):
			raise ValueError("This feature shouldn't have any negative values")

		return final_avgs
	return (name, f)
	














