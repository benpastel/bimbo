import pickle, os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from numpy.random import choice
import datetime

from data import *
from visualize import print_importances
from features import feature_defs

# changing these invalidates the cache
FIT_SAMPLES = 300 * 1000
DEV_SAMPLES = 1000 * 1000

def predict(train, test, clients, products, is_dev):
	tic = datetime.datetime.now()

	defs = feature_defs(clients, products)

	fit_X, fit_Y = generate_fit_features(defs, train, test, FIT_SAMPLES, is_dev)

	toc = datetime.datetime.now()
	feature_time = toc - tic

	model = fit(fit_X, fit_Y)
	print "\tchecking fit error"
	rmse = RMSE(fit_Y, model.predict(fit_X))
	print "\tfit error: %.4f" % rmse
	print_importances(model, defs)

	tic = datetime.datetime.now()
	fit_time = tic - toc

	test_X = generate_test_features(defs, train, test, DEV_SAMPLES, is_dev)

	toc = datetime.datetime.now()
	feature_time += (toc - tic)

	print "predicting"
	preds = model.predict(test_X)

	tic = datetime.datetime.now()
	predict_time = tic - toc

	if np.any(np.isnan(preds)):
		print "WARNING: predict includes %d nans" % np.count_nonzero(np.isnan(preds))

	print "elapsed:"
	print "\t%s feature generation" % feature_time
	print "\t%s model fitting" % fit_time
	print "\t%s model predicting" % predict_time

	print "\nSummary:"
	print "xgboost with default params"
	print "fit rmse: %.4f" % rmse
	print "fit samples: %d" % fit_samples
	print "%d features:" % len(defs)
	print [name for (name, fn) in defs]
	return preds, None

def fit(X, Y):
	print "fitting with XGBoost"
	return xgb.XGBRegressor(subsample=0.1, base_score=np.log(4.0), reg_lambda=2.0).fit(X, Y)

def generate_test_features(feature_defs, train, test, test_samples, is_dev):
	print "generating test features"
	if is_dev:
		print "\tsampling down to %d test samples" % test_samples
		test = test.sample(test_samples, random_state = 1)
	return generate_features(feature_defs, train, test, is_dev, "pickle/dev_features/")

def generate_fit_features(feature_defs, train, test, fit_samples, is_dev):
	print "generating fit features"
	print "\tsplitting train into pool & fit, sampling with fixed seed"
	week8 = (train.week == 8)
	pool, fit = train[~week8], train[week8]
	fit = fit.sample(fit_samples, random_state = 1)
	print "\t%d in pool, %d in model" % (len(pool), len(fit))
	return generate_features(feature_defs, pool, fit, is_dev, "pickle/fit_features/"), fit.net_sales.values

def generate_features(feature_defs, train, test, is_dev, save_dir):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	feats = []
	for name, fn in feature_defs:
		path = save_dir + name
		if is_dev and os.path.isfile(path):
			print "loading feature:", name
			feat = pickle.load(open(path, 'rb'))
		else:
			print "computing feature:", name
			feat = fn(train, test).reshape(-1, 1)
			if is_dev:
				print "\tsaving (%d bytes)" % feat.nbytes
				pickle.dump(feat, open(path, 'wb'))
		feats.append(feat)
	return np.hstack(feats)
	
