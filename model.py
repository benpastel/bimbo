import pickle, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from numpy.random import choice

from data import *
from visualize import print_importances
from features import feature_defs

def predict(train, test, clients, products, is_dev):
	defs = feature_defs(clients, products)
	fit_samples = 10 * 1000

	print "generating features"
	fit_X, fit_Y, test_X, test_Y = generate_features(defs, train, test, fit_samples)

	print "fitting with XGBoost"
	preds, fit_rmse = by_xgboost(fit_X, fit_Y, test_X, test_Y, defs)
	if np.any(np.isnan(preds)):
		print "WARNING: predict includes %d nans" % np.count_nonzero(np.isnan(preds))

	print "\nSummary:"
	print "xgboost with default params"
	print "fit rmse: %d" % fit_rmse
	print "fit samples: %d" % fit_samples
	print "%d features:" % len(features)
	print [name for (name, fn) in features]
	return preds, None

# 
def by_xgboost(fit_X, fit_Y, test_X, test_Y, feature_defs):
	print "\tfit"
	model = xgb.XGBRegressor(nthread=3, subsample=0.1, base_score=np.log(4.0)).fit(fit_X, fit_Y)

	print "\tchecking fit error"
	rmse = RMSE(fit_Y, model.predict(fit_X))
	print "\tfit error: %.4f" % rmse
	print_importances(model, feature_defs)

	return model.predict(test_X), rmse

def generate_features(feature_defs, train, test, fit_samples):
	print "\tsplitting train into pool & fit"
	in_fit = (train.week == 8)
	pool, fit = train[~in_fit], train[in_fit]
	print "\t%d in pool, %d in model" % (len(pool), len(fit))

	print "preparing pool/fit"
	feats = []
	for name, fn in feature_defs:
		print "prepare feature:", name
		feats.append(fn(pool, fit).reshape(-1, 1))
	fit_X = np.hstack(feats)
	fit_Y = fit.net_sales.values

	sample = choice(len(fit_X), fit_samples, replace = False)
	fit_X, fit_Y = fit_X[sample], fit_Y[sample]

	print "preparing train/test"
	feats = []
	for name, fn in feature_defs:
		print "prepare feature:", name
		feats.append(fn(train, test).reshape(-1, 1))
	test_X = np.hstack(feats)
	test_Y = test.net_sales.values

	return fit_X, fit_Y, test_X, test_Y


	
