import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from data import counts_and_avgs, load_data, densify
from visualize import print_importances
from features import feature_defs

def predict(train, test, clients, products, is_dev):
	features = feature_defs(clients, products)

	if is_dev: 
		fit_samples = 100 * 1000
	else: 
		fit_samples = 1000 * 1000

	print "current predictions"
	print "\t%d total:" % len(test)
	preds = by_xgboost(train, test, features, fit_samples)
	if np.any(np.isnan(preds)):
		print "WARNING: predict includes %d nans" % np.count_nonzero(np.isnan(preds))
	return preds, None

def by_xgboost(train, test, features, fit_samples):
	print "training with XGBoost"
	print "\tsplitting train into pool & fit"
	pool = train[train.week < 8]
	fit = train[train.week == 8]
	fit = fit.sample(fit_samples)
	print "\t%d in pool, %d in model" % (len(pool), len(fit))

	print "prepare features"
	fit_feats = []
	for name, fn in features:
		print "feature:", name
		fit_feats.append(fn(pool, fit).reshape(-1, 1))
	X = np.hstack(fit_feats)
	Y = fit.net_sales.values

	print "fit"
	model = xgb.XGBRegressor(nthread=3, subsample=0.1, base_score=np.log(4.0)).fit(X, Y)
	print "\tclassifier: %s" % model

	print "checking error on fit data"
	fit_Y = model.predict(X)
	rmse = np.sqrt( np.average( (Y - fit_Y)**2 ) )
	print "fit error: %.4f" % rmse
	del X, Y, fit_Y
	print_importances(model, features)

	print "saving to file"
	with open("pickle/model.pickle", 'w') as f:
		pickle.dump(model, f)

	print "predicting"
	test_feats = []
	for name, fn in features:
		print "feature:", name
		test_feats.append(fn(train, test).reshape(-1, 1))
	test_X = np.hstack(test_feats)
	return model.predict(test_X)

# def by_linear_regression(train, test, clients):
# 	print "training linear regression"
# 	features = {
# 		"product_client avgs": lambda train, test: avg_by_key(train, test, product_client_hash),
# 		"clientname_product avgs": lambda train, test: client_name_features(train, test, clients),
# 		"product factors": lambda train, test: product_factor_vs_client_features(train, test),
# 		"client factors": lambda train, test: client_factor_vs_product_features(train, test)
# 	}

# 	print "\tsplitting train into pool & model"
# 	pool = train[train.week < 8]
# 	model = train[train.week == 8]
# 	model = model.sample(1000 * 1000)
# 	print "\t%d in pool, %d in model" % (len(pool), len(model))

# 	print "prepare features"
# 	model_feats = []
# 	for name, fn in features.iteritems():
# 		model_feats.append(fn(pool, model).reshape(-1, 1))
# 	X = np.hstack(model_feats)

# 	nans = np.isnan(X)
# 	print "\t (using median val for %d nan features)" % np.count_nonzero(nans)
# 	X[nans] = np.log(4.0)

# 	Y = model.net_sales.values.reshape(-1, 1)

# 	print "\tfit"
# 	print "\t\tshapes: %s X, %s Y" % (X.shape, Y.shape)
# 	regression = LinearRegression().fit(X, Y)
# 	theta = regression.coef_[0]
# 	b = regression.intercept_
# 	print "\tfound theta=%s, b=%.6f" % (theta, b)

# 	print "regenerating features with full training set and predicting"
# 	i = 0
# 	preds = np.full(len(test), b, dtype=np.float32)
# 	for name, fn in features.iteritems():
# 		test_feat = fn(train, test)
# 		nans = np.isnan(test_feat)
# 		print "\t (%s: using median val for %d nans)" % (name, np.count_nonzero(nans))
# 		test_feat[nans] = np.log(4.0)
# 		preds += test_feat * theta[i]
# 		i += 1
# 	return preds



	
