import numpy as np
import pandas as pd
import os
import csv

from models import *
from data import load_data, RMSLE

def rmsle_breakdown_by_count(preds, actuals, counts):
	for c in range(10):
		matches = (counts == c)
		hits = np.sum(matches)
		rmsle = RMSLE(preds[matches], actuals[matches])
		print "count %d: %d points, %0.4f RMSLE" % (c, hits, rmsle)

	matches = (counts >= 10)
	hits = np.sum(matches)
	rmsle = RMSLE(preds[matches], actuals[matches])
	print "count >= 10: %d points, %0.4f RMSLE" % (hits, rmsle)

def predict_dev():
	model_fns = [
		# avg_pair_avg_product_factors,
		# logavg_pair_logavg_product_factors,
		reference,
		current
	]
	train, dev, test, clients, products = load_data(dev_sample=1000 * 1000)

	for model_fn in model_fns:
		print "making dev predictions with " + str(model_fn) + "..."
		preds, counts = model_fn(train, dev, clients, products, "for_dev")
		print "total RMSLE: ", RMSLE(preds, dev.sales)
		rmsle_breakdown_by_count(preds, dev.sales, counts)

def predict_test():
	model_fn = logavg_pair_logavg_product_factors
	train, dev, test, clients, products = load_data(dev_sample=None)

	print "making test predictions with " + str(model_fn) + "..."
	preds, _ = model_fn(pd.concat([train, dev]), test, "for_test")
	if len(preds) != len(test): raise Exception("wrong prediction length")

	print "writing predictions to file"
	test["predictions"] = preds
	test.to_csv("pred/log_product_factors.csv", header = False, columns = ("id", "predictions"), index = False)

if __name__ == "__main__":
	predict_dev()
	# predict_test()





