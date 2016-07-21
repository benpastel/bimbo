import numpy as np
import pandas as pd
import os
import csv
import sys

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
		# reference,
		current
	]
	train, dev, test, clients, products = load_data()

	for model_fn in model_fns:
		print "making dev predictions with " + str(model_fn) + "..."
		preds, counts = model_fn(train, dev, clients, products, "for_dev")
		print "total RMSLE: ", RMSLE(preds, dev.sales)
		if counts:
			rmsle_breakdown_by_count(preds, dev.sales, counts)

def predict_test():
	model_fn = current
	train, dev, test, clients, products = load_data()

	print "making test predictions with " + str(model_fn) + "..."
	preds, _ = model_fn(pd.concat([train, dev]), test, clients, products, "for_test")
	if len(preds) != len(test): raise Exception("wrong prediction length")

	print "writing predictions to file"
	test["predictions"] = preds
	test.to_csv("pred/output.csv", header = False, columns = ("id", "predictions"), index = False)

if __name__ == "__main__":
	if len(sys.argv) != 2 or sys.argv[1] not in {"dev", "test"}:
		raise ValueError("usage: %s [dev|test]" % sys.argv[0])
	if sys.argv[1] == "dev":
		predict_dev()
	else:
		predict_test()





