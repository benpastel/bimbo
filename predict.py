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
		predict_avg_with_price_factors
	]
	train, dev, _ = load_data(dev_sample=1000 * 1000)

	for model_fn in model_fns:
		print "making dev predictions with " + str(model_fn) + "..."
		preds, counts = model_fn(train, dev, "for_dev")
		print "total RMSLE: ", RMSLE(preds, dev["net_units_sold"])
		rmsle_breakdown_by_count(preds, dev["net_units_sold"], counts)

def predict_test():
	model_fn = log_avg_product_factors
	train, dev, test = load_data(dev_sample=None)

	# TODO: re-run test on train + dev!

	print "making test predictions with " + str(model_fn) + "..."
	preds, _ = model_fn(pd.concat(train, dev), test)

	# merge predictions with test ids

	# write predictions to file


if __name__ == "__main__":
	predict_dev()





