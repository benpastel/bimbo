import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from data import counts_and_avgs, load_data, densify
from product_factors import *

def by_xgboost(train, test):
	print "training with XGBoost"

	train_X = train[["client_id", "product_id", "depot_id"]].sample(n=1 * 1000 * 1000).as_matrix()
	train_Y = train["sales"]
	test_X = test[["client_id", "product_id", "depot_id"]].as_matrix()

	model = xgb.XGBClassifier(subsample=0.1).fit(train_X, train_Y)
	return model.predict(test_X)

def product_client_hash(frame):
	return frame.client_key.values.astype(np.int64) * 3000 + frame.product_key.values

def product_client_depot_hash(frame):
	return (frame.client_key.values * (3000 * 600)
		+ frame.product_key.values * 600
		+ frame.depot_key.values)

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

	print "\tprepare features"
	model_feats = []
	for name, fn in features.iteritems():
		_, model_feat = fn(pool, model)
		model_feats.append(model_feat.reshape(-1, 1))
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
		_, test_feat = fn(train, test)
		nans = np.isnan(test_feat)
		print "\t (%s: using median val for %d nans)" % (name, np.count_nonzero(nans))
		test_feat[nans] = np.log(4.0)
		preds += test_feat * theta[i]
		i += 1

	print "\texp"
	return np.exp(preds) - 1

def avg_by_key(train, test, key_fn):
	print "\tbuilding keys"
	train_keys, test_keys = densify(key_fn(train), key_fn(test))
	max_key = max(np.max(train_keys), np.max(test_keys))

	print "\tpooling log means"
	_, means = counts_and_avgs(train_keys, train.log_sales.values, max_group = max_key)

	return means[train_keys], means[test_keys]

def client_name_features(train, test, clients):
	print "\tfeatures on (client_name, product)"
	print "\tbuilding common client keys"
	train_client_key, test_client_key, client_client_key = densify(
		train.client_id.values, 
		test.client_id.values, 
		clients.index.values)
	
	print "\thashing client names"
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
	return means[train_keys], means[test_keys]

def product_factor_vs_client_features(train, test):
	print "features (avg multiplier for product) * (client avg)"
	return avg_factor_features(train, test, 
		lambda frame: frame.client_key.values,
		lambda frame: frame.product_key.values)



	
