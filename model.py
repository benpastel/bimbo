import pickle, os, sys
import numpy as np
import pandas as pd
from numpy.random import choice
import datetime

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, SGDRegressor, LassoCV

from data import *
from visualize import print_importances
from features import feature_defs

# TODO: actually keep the subsample in? did worse on full set
L1_MODELS = [
	("xgb short wide", XGBRegressor(base_score=np.log(4.0), reg_lambda=2.0, min_child_weight=10, 
		max_depth=2, n_estimators=200)), 
	("xgb tall skinny", XGBRegressor(base_score=np.log(4.0), reg_lambda=2.0, min_child_weight=10, 
		max_depth=5, n_estimators=30)),
	("xgb no subsample", XGBRegressor(base_score=np.log(4.0), reg_alpha=2.0)),
	("xgb subsample", XGBRegressor(subsample=0.5, base_score=np.log(4.0), reg_alpha=2.0)),
]
L2_MODEL = LassoCV(positive=True)

def predict(train, test, clients, products, is_dev):
	tic = datetime.datetime.now()

	# changing these invalidates feature caches
	fit_samples = 1000 * 1000 if is_dev else 5000 * 1000
	dev_samples = 1000 * 1000

	defs = feature_defs(clients, products)

	fit_X, fit_Y = generate_fit_features(defs, train, test, fit_samples, is_dev)

	toc = datetime.datetime.now()
	feature_time = toc - tic

	L1s, L2 = fit(fit_X, fit_Y)
	tic = datetime.datetime.now()
	fit_time = tic - toc

	print "\tchecking fit error"
	rmse = RMSE(fit_Y, model_predict(fit_X, L1s, L2))
	print "\tfit error: %.4f" % rmse
	describe(L1s, L2)

	test_X, test_Y = generate_test_features(defs, train, test, dev_samples, is_dev)

	toc = datetime.datetime.now()
	feature_time += (toc - tic)

	print "predicting"
	del fit_X, fit_Y, train, test # prediction needs all the memory it can get
	preds = model_predict(test_X, L1s, L2)

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
	return preds, test_Y

def model_predict(X, L1s, L2):
	outs = []
	for i, L1 in enumerate(L1s):
		print "\tL1 predicting", L1_MODELS[i][0]
		outs.append(L1.predict(X).reshape(-1, 1))
	inter = np.hstack(outs)
	print "\tL2 prediction"
	return L2.predict(inter)

def fit(X, Y):
	model_count = len(L1_MODELS) + 1
	Xs = split(model_count, X)
	Ys = split(model_count, Y)
	L1s = []

	for i, (name, model) in enumerate(L1_MODELS):
		X = Xs[i]
		Y = Ys[i]
		print "L1: fitting %d points with %s" % (len(X), name)
		L1s.append(model.fit(X, Y))

	print "L2: fitting %d points with %s" 
	X = Xs[-1]
	Y = Ys[-1]
	outs = []
	for i, L1 in enumerate(L1s):
		out = L1.predict(X).reshape(-1, 1)
		print "\t%s output: max %.1f, min %.1f, avg %.1f" % (L1_MODELS[i][0], np.max(out), np.min(out), np.average(out))
		if np.any(np.isnan(out)): raise ValueError("L1 model output a nan")
		if np.any(np.isinf(out)): raise ValueError("L1 model output an inf")
		outs.append(out)
	inter = np.hstack(outs)
	L2 = L2_MODEL.fit(inter, Y)

	return L1s, L2

def generate_test_features(feature_defs, train, test, test_samples, is_dev):
	print "generating test features"
	if is_dev:
		print "\tsampling down to %d test samples" % test_samples
		test = test.sample(test_samples, random_state = 1)
		test_Y = test.net_sales.values
		path = "pickle/dev_features/"
	else:
		test_Y = None
		path = "pickle/test_features/"

	return generate_features(feature_defs, train, test, is_dev, path), test_Y

def generate_fit_features(feature_defs, train, test, fit_samples, is_dev):
	print "generating fit features"
	print "\tsplitting train into pool & fit, sampling with fixed seed"
	last_week = np.max(train.week.values)
	is_last_week = (train.week == last_week)
	pool, fit = train[~is_last_week], train[is_last_week]
	fit = fit.sample(fit_samples, random_state = 1)
	print "\t%d for pooling, %d for fitting (fit week: %d)" % (len(pool), len(fit), last_week)

	if is_dev:
		path = "pickle/fit_for_dev_features/"
	else:
		path = "pickle/fit_for_test_features/"
	return generate_features(feature_defs, pool, fit, is_dev, path), fit.net_sales.values

def generate_features(feature_defs, train, test, is_dev, save_dir):
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	feats = []
	for name, fn in feature_defs:
		path = save_dir + name
		if os.path.isfile(path):
			print "loading feature:", name
			feat = pickle.load(open(path, 'rb'))
		else:
			print "computing feature:", name
			feat = fn(train, test).reshape(-1, 1)
			
			print "\tsaving (%d mb)" % (feat.nbytes / 1000000)
			pickle.dump(feat, open(path, 'wb'))
		feats.append(feat)
	X = np.hstack(feats)
	nans = np.isnan(X)
	print "\t replacing %d NaNs with 0" % np.count_nonzero(nans)
	X[nans] = 0
	return X

def describe(L1s, L2):
	print "L2 coefficients:"
	for i, (name, _) in enumerate(L1_MODELS):
		print "\t%s: %.4f" % (name, L2.coef_[i])

	print "b:", L2.intercept_

def split(ways, M):
	slices = []
	start = 0
	c = len(M) / ways
	for i in range(ways):
		slices.append(M[start:start+c])
		start += c
	return slices

	
