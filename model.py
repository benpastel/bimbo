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
	fit_samples = 1000 * 1000

	print "generating fit features"
	fit_X, fit_Y = generate_fit_features(defs, train, test, fit_samples)

	print "fitting with XGBoost"
	model = fit(fit_X, fit_Y)
	print "\tchecking fit error"
	rmse = RMSE(fit_Y, model.predict(fit_X))
	print "\tfit error: %.4f" % rmse
	print_importances(model, defs)

	print "generating test features"
	test_X = generate_test_features(defs, train, test)

	print "predicting"
	preds = model.predict(test_X)

	if np.any(np.isnan(preds)):
		print "WARNING: predict includes %d nans" % np.count_nonzero(np.isnan(preds))

	print "\nSummary:"
	print "xgboost with default params"
	print "fit rmse: %.4f" % rmse
	print "fit samples: %d" % fit_samples
	print "%d features:" % len(defs)
	print [name for (name, fn) in defs]
	return preds, None

def fit(X, Y):
	return xgb.XGBRegressor(nthread=3, subsample=0.1, base_score=np.log(4.0)).fit(X, Y)

def generate_test_features(feature_defs, train, test):
	feats = []
	for name, fn in feature_defs:
		print "prepare feature:", name
		feats.append(fn(train, test).reshape(-1, 1))
	return np.hstack(feats)

def generate_fit_features(feature_defs, train, test, fit_samples):
	print "\tsplitting train into pool & fit, sampling"
	week8 = (train.week == 8)
	pool, fit = train[~week8], train[week8]
	fit = fit.sample(fit_samples)
	print "\t%d in pool, %d in model" % (len(pool), len(fit))

	feats = []
	for name, fn in feature_defs:
		print "prepare feature:", name
		feats.append(fn(pool, fit).reshape(-1, 1))
	return np.hstack(feats), fit.net_sales.values


	
