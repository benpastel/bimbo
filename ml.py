import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from data import counts_and_avgs, load_data, densify
from product_factors import *

def last_nonzero_logsale(train, test):
	key_fn = product_client_hash
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	by_key = np.zeros(max_key + 1, dtype = np.int32)

	test_week = extract_week(test)
	for week in range(3, test_week):
		# overwrite values with the most recent week
		ok = (train.week.values == week) & (train.log_sales.values > 0)
		k = train_keys[ok]
		by_key[k] = train.log_sales.values[ok]
	return by_key[test_keys]

# TODO: try treating different test data differently
def extract_week(test):
	return test.week.values[0]
	# if "week" in test.columns:
	# 	out = test.week.values[0]
	# else:
	# 	out = 10
	# print "(pretending all of test is week %d)" % out
	# return out

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
	vals[train_keys] = last_train.log_sales.values
	out = vals[test_keys]
	print "\t%d/%d nonzero vals" % (np.count_nonzero(out), len(test))
	return out

def by_xgboost(train, test, clients):
	print "training with XGBoost"

	features = [
		("last_nonzero_logsale", last_nonzero_logsale),
		("weeks_since_last_sale", weeks_since_last_sale),
		("logsale_last_week", logsale_last_week),
		("product_client avgs", lambda train, test: avg_by_key(train, test, product_client_hash)),
		("clientname_product avgs", lambda train, test: client_name_features(train, test, clients)),
		("product factors", lambda train, test: product_factor_vs_client_features(train, test)),
		("depot_avg", lambda train, test: avg_by_key(train, test, lambda frame:frame.depot_key)),
		("product_avg", lambda train, test: avg_by_key(train, test, lambda frame:frame.product_key)),
		("client_avg", lambda train, test: avg_by_key(train, test, lambda frame:frame.client_key)),
		("product_depot_avg", lambda train, test: avg_by_key(train, test, product_depot_hash)),
	]
	print "\tsplitting train into pool & fit"
	pool = train[train.week < 8]
	fit = train[train.week == 8]
	fit = fit.sample(1000 * 1000)
	print "\t%d in pool, %d in model" % (len(pool), len(fit))

	print "prepare features"
	fit_feats = []
	for name, fn in features:
		print "feature:", name
		fit_feats.append(fn(pool, fit).reshape(-1, 1))
	X = np.hstack(fit_feats)
	Y = fit.log_sales.values

	print "fit"
	model = xgb.XGBRegressor(nthread=3, subsample=0.1, base_score=np.log(4.0)).fit(X, Y)
	print "\tclassifier: %s" % model

	print "checking error on fit data"
	fit_Y = model.predict(X)
	rmse = np.sqrt( np.average( (Y - fit_Y)**2 ) )
	print "error: %.4f" % rmse
	del X, Y, fit_Y

	print "predicting"
	test_feats = []
	for name, fn in features:
		print "feature:", name
		test_feats.append(fn(train, test).reshape(-1, 1))
	test_X = np.hstack(test_feats)
	preds = model.predict(test_X)
	return np.exp(preds) - 1

def product_client_hash(frame):
	return frame.client_key.values.astype(np.int64) * 3000 + frame.product_key.values

def product_client_depot_hash(frame):
	return (frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

def product_depot_hash(frame):
	return frame.product_key.values * 600 + frame.depot_key.values

def by_linear_regression(train, test, clients):
	print "training linear regression"
	features = {
		"product_client avgs": lambda train, test: avg_by_key(train, test, product_client_hash),
		"clientname_product avgs": lambda train, test: client_name_features(train, test, clients),
		"product factors": lambda train, test: product_factor_vs_client_features(train, test),
	}

	print "\tsplitting train into pool & model"
	pool = train[train.week < 8]
	model = train[train.week == 8]
	model = model.sample(1000 * 1000)
	print "\t%d in pool, %d in model" % (len(pool), len(model))

	print "prepare features"
	model_feats = []
	for name, fn in features.iteritems():
		model_feats.append(fn(pool, model).reshape(-1, 1))
	X = np.hstack(model_feats)

	nans = np.isnan(X)
	print "\t (using median val for %d nan features)" % np.count_nonzero(nans)
	X[nans] = np.log(4.0)

	Y = model.log_sales.values.reshape(-1, 1)

	print "\tfit"
	print "\t\tshapes: %s X, %s Y" % (X.shape, Y.shape)
	regression = LinearRegression().fit(X, Y)
	theta = regression.coef_[0]
	b = regression.intercept_
	print "\tfound theta=%s, b=%.6f" % (theta, b)

	print "regenerating features with full training set and predicting"
	i = 0
	preds = np.full(len(test), b, dtype=np.float32)
	for name, fn in features.iteritems():
		test_feat = fn(train, test)
		nans = np.isnan(test_feat)
		print "\t (%s: using median val for %d nans)" % (name, np.count_nonzero(nans))
		test_feat[nans] = np.log(4.0)
		preds += test_feat * theta[i]
		i += 1
	return np.exp(preds) - 1

def avg_by_key(train, test, key_fn):
	print "\tbuilding keys"
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	print "\tpooling log means"
	_, means = counts_and_avgs(train_keys, train.log_sales.values, max_group = max_key)

	return means[test_keys]

def client_name_features(train, test, clients):
	train_client_key, test_client_key, client_client_key = densify(
		train.client_id.values, 
		test.client_id.values, 
		clients.index.values)
	
	print "\t\thashing client names"
	client_hashes = np.zeros(np.max(client_client_key) + 1, dtype=np.int64)
	for r, name in enumerate(clients.client_name):
		client_key = client_client_key[r]
		client_hashes[client_key] = hash(name)

	def key(frame, frame_client_keys):
		return (
			client_hashes[frame_client_keys] * 3000
			+ frame.product_key.values)
	train_keys, test_keys = densify(
		key(train, train_client_key),
		key(test, test_client_key))

	_, means = counts_and_avgs(train_keys, train.log_sales.values)
	return means[test_keys]

def product_factor_vs_client_features(train, test):
	return avg_factor_features(train, test, 
		lambda frame: frame.client_key.values,
		lambda frame: frame.product_key.values)



	
