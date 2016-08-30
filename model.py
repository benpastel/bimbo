import pickle, os, sys
import numpy as np
import pandas as pd
from numpy.random import choice
import datetime

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, SGDRegressor, LassoCV
from sklearn import preprocessing

# import tensorflow as tf

from data import *
from visualize import print_importances
# from partitioning import partition_feature_defs
from features import feature_defs

IMPUTE_VALUE = -1
MAX_DEV_SAMPLES_PER_MODEL = 100 * 1000

# class NN:
# 	def __init__(self):
# 		self.net = tf.contrib.learn.DNNRegressor(hidden_units=[20, 5], dropout=0.1)

# 	def fit(self, X, Y):
# 		self.net = self.net.fit(X, Y, steps=1000)
# 		return self

# 	def predict(self, X):
# 		return self.net.predict(X, batch_size=100)

# kicking up the n_estimator seems to be helpful, but slows things down a lot
L1_MODELS = [
	# ("Neural Net", NN()),
	("xgb shallow fat", XGBRegressor(subsample=0.95, base_score=np.log(4.0), reg_lambda=0.0, reg_alpha=10.0, min_child_weight=60, 
		max_depth=3, n_estimators=500)),
	("xgb deep skinny", XGBRegressor(subsample=0.95, base_score=np.log(4.0), reg_lambda=0.0, reg_alpha=40.0, min_child_weight=100, 
		max_depth=12, n_estimators=50)),
	("xgb middle", XGBRegressor(subsample=0.95, base_score=np.log(4.0), reg_lambda=0.0, reg_alpha=10.0, min_child_weight=60, 
		max_depth=5, n_estimators=200)),
]
L2_MODEL = LassoCV()

def predict(train, test, clients, products, is_dev):
	tic = datetime.datetime.now()

	# changing these invalidates feature caches
	fit_samples = 1000 * 1000 if is_dev else 5000 * 1000
	dev_samples = 1000 * 1000

	defs = feature_defs(clients, products)
	# defs = partition_feature_defs(clients, products)

	fit_X, fit_Y = generate_fit_features(defs, train, test, fit_samples, is_dev)
	if is_dev:
		# number of L1 models might have dropped since we cached features
		# make sure we don't go too slow
		max_n = MAX_DEV_SAMPLES_PER_MODEL * (len(L1_MODELS) + 1)
		fit_X = fit_X[0:max_n,:]
		fit_Y = fit_Y[0:max_n]

	toc = datetime.datetime.now()
	feature_time = toc - tic

	L1s, L2, rmse = fit(fit_X, fit_Y, defs)
	tic = datetime.datetime.now()
	fit_time = tic - toc

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

def fit(X, Y, feature_defs):
	model_count = len(L1_MODELS) + 1
	Xs = split(model_count, X)
	Ys = split(model_count, Y)
	L1s = []
	outs = []

	for i, (name, model) in enumerate(L1_MODELS):
		X = Xs[i]
		Y = Ys[i]
		print "L1: fitting %d points with %s" % (len(X), name)
		fitted = model.fit(X, Y)
		L1s.append(fitted)
		print "\tpredict"

		fit_out = fitted.predict(X)
		print "\tRMSE: %.3f fit" % RMSE(Y, fit_out)

		lastX_out = fitted.predict(Xs[-1])
		print "\tRMSE: %.3f holdout in same week" % RMSE(Ys[-1], lastX_out)
		outs.append(lastX_out.reshape(-1, 1))

	X = Xs[-1]
	Y = Ys[-1]
	print "L2: fitting %d points with %s" % (len(X), L2_MODEL)
	inter = np.hstack(outs)
	L2 = L2_MODEL.fit(inter, Y)
	fit_out = L2.predict(inter)
	rmse = RMSE(Y, fit_out)
	print "\tRMSE: %.3f fit" % rmse

	print "Feature importances of 1st model:"
	print print_importances(L1s[0], feature_defs)
	
	print "L2 coefficients:"
	for i, (name, _) in enumerate(L1_MODELS):
		print "\t%s: %.4f" % (name, L2.coef_[i])
	print "b:", L2.intercept_

	return L1s, L2, rmse

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
	print "\t replacing %d NaNs with %d" % (np.count_nonzero(nans), IMPUTE_VALUE)
	X[nans] = IMPUTE_VALUE
	print "\tscaling..."
	X = preprocessing.MinMaxScaler().fit_transform(X)
	return X

def split(ways, M):
	c = M.shape[0] / ways
	slices = []
	start = 0
	for i in range(ways):
		slices.append(M[start:start+c])
		start += c
	return slices

	
