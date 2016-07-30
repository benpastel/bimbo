import pickle, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from numpy.random import choice

from data import *
from visualize import print_importances
from features import feature_defs

DEV_FIT_PATH = "pickle/dev_fit_X"
DEV_TEST_PATH = "pickle/dev_test_X"
DEV_MODEL_PATH = "pickle/dev_model"

TEST_FIT_PATH = "pickle/test_fit_X"
TEST_TEST_PATH = "pickle/test_test_X"
TEST_MODEL_PATH = "pickle/test_model"

def predict(train, test, clients, products, is_dev):
	defs = feature_defs(clients, products)
	regen_features = True
	regen_model = True
	fit_samples = 1000 * 1000

	print "generating features"
	fit_X, fit_Y, test_X, test_Y = generate_features(defs, train, test, regen_features, is_dev, fit_samples)

	print "fitting with XGBoost"
	preds, fit_rmse = by_xgboost(fit_X, fit_Y, test_X, test_Y)
	if np.any(np.isnan(preds)):
		print "WARNING: predict includes %d nans" % np.count_nonzero(np.isnan(preds))

	print "\nSummary:"
	print "xgboost with default params"
	print "fit rmse: %d" % fit_rmse
	print "fit samples: %d" % fit_samples
	print "%d features:" % len(features)
	print [name for (name, fn) in features]
	return preds, None

def by_xgboost(fit_X, fit_Y, test_X, test_Y, regen, is_dev):
	path = (DEV_MODEL_PATH if is_dev else TEST_MODEL_PATH)

	if regen:
		print "\tfit"
		model = xgb.XGBRegressor(nthread=3, subsample=0.1, base_score=np.log(4.0)).fit(X, Y)

		print "\tsaving model to file"
		with open(path, 'w') as f:
			pickle.dump(model, f)
	else:
		print "loading", path
		with open(path, 'r') as f:
			model = pickle.load(f)

	print "\tchecking fit error"
	rmse = RMSE(fit_Y, model.predict(X))
	print "\tfit error: %.4f" % rmse
	print_importances(model, features)

	return model.predict(test_X), rmse

def generate_features(feature_defs, train, test, regen, is_dev, fit_samples):
	fit_path = (DEV_FIT_PATH if is_dev else TEST_FIT_PATH)
	test_path = (DEV_TEST_PATH if is_dev else TEST_TEST_PATH)

	print "\tsplitting train into pool & fit"
	in_fit = (train.week == 8)
	pool, fit = train[~in_fit], train[in_fit]
	print "\t%d in pool, %d in model" % (len(pool), len(fit))

	if regen:
		print "preparing pool/fit"
		feats = []
		for name, fn in feature_defs:
			print "prepare feature:", name
			feats.append(fn(pool, fit).reshape(-1, 1))
		fit_X = np.hstack(feats)
		fit_Y = fit.net_sales.values

		sample = choice(len(fit_X), fit_samples, replace = False)
		fit_X, fit_Y = fit_X[sample], fit_Y[sample]
		print "\tsaving fit features"
		with open(fit_path, 'w') as f:
			pickle.dump((fit_X, fit_Y), f)

	else:
		print "loading fit", fit_path, test_path
		with open(fit_path, 'r') as f:
			fit_X, fit_Y = pickle.load(f)

	print "preparing train/test"
	feats = []
	for name, fn in feature_defs:
		print "prepare feature:", name
		feats.append(fn(train, test).reshape(-1, 1))
	test_X = np.hstack(feats)
	test_Y = test.net_sales.values

	print "\tsaving test features"
	with open(test_path, 'w') as f:
		pickle.dump((test_X, test_Y), f)

	return fit_X, fit_Y, test_X, test_Y


	
