import numpy as np
import pandas as pd
import random

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

	feats = []
	random.seed(1)
	possible_levels = all_key_fns.keys()

	for trial in range(10):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 100))

	for trial in range(10):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 300))

	for trial in range(10):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 1000))
	return feats

def make_partition_feature(level_order, all_key_fns, max_group_size):
	name = "partition_" + str(max_group_size) + "_" + "_".join(level_order[0:MAX_PARTITION_LEVELS])

	def f(train, test):
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

			# extract the latest key
			key_fn = all_key_fns[level]
			new_train_keys = is_train_unfinished * key_fn(train)
			new_test_keys = is_test_unfinished * key_fn(test)

			# TODO: probably need this densify...
			# new_train_keys, new_test_keys = densify(new_train_keys, new_test_keys)
		
			# shift previous keys out of the way
			shift = max(np.max(new_train_keys), np.max(new_test_keys)) + 1
			# print "\t\tshift:", shift
			train_keys = train_keys.astype(np.int64) * shift + (new_train_keys + 1)
			test_keys = test_keys.astype(np.int64) * shift + (new_test_keys + 1)

			if np.any(train_keys < 0) or np.any(test_keys < 0):
				print train_keys.dtype, test_keys.dtype
				raise Exception("Overflow! Guess we need densify() after all :(")

			# check whether the unfinished groups are now small enough by collapsing all finished keys to 0
			train_keys, test_keys = densify(train_keys, test_keys)
			max_key = max(np.max(train_keys), np.max(test_keys))
			counts, avgs = counts_and_avgs(train_keys, train.net_sales.values, max_key)

			new_group_unfinished = (counts > max_group_size)
			# print "\t\t%d/%d new groups unfinished (+/- 1)" % (np.count_nonzero(new_group_unfinished), len(new_group_unfinished))

			test_is_nan = (counts[test_keys] == 0)
			# print "\t\t%d/%d test keys became nan; keeping previous estimate (or 0)" % (np.count_nonzero(test_is_nan), len(test))
			is_test_unfinished &= ~test_is_nan

			final_avgs[is_test_unfinished] = avgs[test_keys][is_test_unfinished]

			is_train_unfinished &= new_group_unfinished[train_keys]
			is_test_unfinished &= new_group_unfinished[test_keys]

			train_keys *= is_train_unfinished 		
			test_keys *= is_test_unfinished

			# print "\t\t%d/%d trains unfinished" % (np.count_nonzero(is_train_unfinished), len(train))
			# print "\t\t%d/%d tests unfinished" % (np.count_nonzero(is_test_unfinished), len(test))

		# print "\t\t%d/%d final_avgs are nonzero" % (np.count_nonzero(final_avgs), len(final_avgs))
		if np.any(np.isnan(final_avgs)): 
			raise ValueError("This feature shouldn't have any NaN values")
		if np.any(final_avgs < 0):
			raise ValueError("This feature shouldn't have any negative values")

		return final_avgs
	return (name, f)













