import numpy as np
import pandas as pd
import os
import csv
import sys

from data import load_data, RMSE
from model import predict

def rmsle_breakdown_by_count(preds, actuals, counts):
	for c in range(10):
		matches = (counts == c)
		hits = np.sum(matches)
		rmsle = RMSE(preds[matches], actuals[matches])
		print "count %d: %d points, %0.4f RMSLE" % (c, hits, rmsle)

	matches = (counts >= 10)
	hits = np.sum(matches)
	rmsle = RMSE(preds[matches], actuals[matches])
	print "count >= 10: %d points, %0.4f RMSLE" % (hits, rmsle)

def predict_dev():
	train, dev, test, clients, products = load_data()

	preds, Y = predict(train, dev, clients, products, is_dev=True)
	print "total RMSLE: %.4f" % RMSE(preds, Y)

def predict_test():
	train, dev, test, clients, products = load_data()

	preds, _ = predict(pd.concat([train, dev]), test, clients, products, is_dev=False)
	if len(preds) != len(test): raise Exception("wrong prediction length")

	print "writing predictions to file"
	test["predictions"] = np.exp(preds) - 1
	test.to_csv("pred/output.csv", header = False, columns = ("id", "predictions"), index = False)

if __name__ == "__main__":
	if len(sys.argv) != 2 or sys.argv[1] not in {"dev", "test"}:
		raise ValueError("usage: %s [dev|test]" % sys.argv[0])
	if sys.argv[1] == "dev":
		predict_dev()
	else:
		predict_test()





