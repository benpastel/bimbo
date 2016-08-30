import numpy as np
import pandas as pd
import random

from data import *

# TODO: this will be circular dependency
from features import as_fn, clientname_hash_fn

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
	for trial in range(33):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 1))

	for trial in range(33):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 10))

	for trial in range(33):
		random.shuffle(possible_levels)
		feats.append(make_partition_feature([l for l in possible_levels], all_key_fns, 100))

	return feats

def make_partition_feature(level_order, all_key_fns, max_group_size):
	name = "partition_" + str(max_group_size) + "_" + "_".join(level_order)

	def f(train, test):
		is_train_finished = np.zeros((len(train), ), dtype=np.bool)
		is_test_finished = np.zeros((len(test), ), dtype=np.bool)
		final_avgs = np.zeros((len(test), ), dtype=np.float32)

		level_count = 0
		levels = []
		while (level_count < len(level_order) 
			and np.count_nonzero(is_train_finished) < len(train) 
			and np.count_nonzero(is_test_finished) < len(test)):
		
			level = level_order[level_count]
			level_count += 1
			levels.append(level)

			print "\tpartioning on", level

			# re-densify all previous keys on the unfinished data.
			unfinished_train = train[~is_train_finished]
			unfinished_test = test[~is_test_finished]
			
			# print "\t%d/%d unfinished train" % (len(unfinished_train), len(train))
			# print "\t%d/%d unfinished test" % (len(unfinished_test), len(test))

			# combine all the key functions so far
			key_fns = [all_key_fns[l] for l in levels]
			max_keys = []
			for key_fn in key_fns:
				train_keys, test_keys = densify(key_fn(unfinished_train), key_fn(unfinished_test))
				max_keys.append( max(np.max(train_keys), np.max(test_keys)) )
			combined_fn = combine(key_fns, max_keys)

			# apply the combined functions
			train_keys, test_keys = densify(combined_fn(unfinished_train), combined_fn(unfinished_test))
			max_key = max(np.max(train_keys), np.max(test_keys))
			counts, avgs = counts_and_avgs(train_keys, unfinished_train.net_sales.values, max_group = max_key)

			# print "\tmax key:", max_key

			# check resulting sizes
			new_group_finished = (counts <= max_group_size)

			# print "\t%d/%d new groups finished (< max size)" % (np.count_nonzero(new_group_finished), len(new_group_finished))

			# these masks are indexed on the unfinished data
			new_train_finished = new_group_finished[train_keys]
			new_test_finished = new_group_finished[test_keys]

			# print "\t%d/%d trains just finished" % (np.count_nonzero(new_train_finished), len(new_train_finished))
			# print "\t%d/%d tests just finished" % (np.count_nonzero(new_test_finished), len(new_test_finished))

			# if a test key has count 0, then we went too deep... we actually want to use the estimate on the previous level.
			# n.b. we could just keep going deeper until this happens to everyone!
			#
			new_test_is_nan = (counts[test_keys] == 0)
			# print "\t%d/%d test keys became nan at this level; will lock in previous estimate (or 0)" % (
				# np.count_nonzero(new_test_is_nan), len(test_keys))

			# add the newly-finished avgs to original indices
			mask = avgs[test_keys]
			mask[~new_test_finished] = 0 # don't update anything that's not finished
			mask[new_test_is_nan] = 0 # don't update anything that became NaN
			final_avgs[~is_test_finished] += mask

			# print "\t%d/%d final_avgs are nonzero" % (np.count_nonzero(final_avgs), len(final_avgs))

			# update masks for the original indices
			is_train_finished[~is_train_finished] = new_train_finished
			is_test_finished[~is_test_finished] = new_test_finished

		# print "all done!"
		# unfinished_train = train[~is_train_finished] # TODO: remove
		# unfinished_test = test[~is_test_finished] # TODO: remove
		# print "\t%d/%d unfinished train" % (len(unfinished_train), len(train))
		# print "\t%d/%d unfinished test" % (len(unfinished_test), len(test)) 
		# print "\t%d/%d final_avgs are nonzero" % (np.count_nonzero(final_avgs), len(final_avgs))
		if np.any(np.isnan(final_avgs)): 
			raise ValueError("This feature shouldn't have any NaN values")

		return final_avgs
	return (name, f)

def combine(key_fns, max_keys):
	def fn(frame):
		# apply the first key
		vals = key_fns[0](frame).astype(np.int64)

		for i in range(1, len(key_fns)):

			# shift the result so far out of the way
			vals = (vals + 1) * (max_keys[i] + 1)

			# add the next key
			vals += key_fns[i](frame).astype(np.int64)
		return vals

	return fn













