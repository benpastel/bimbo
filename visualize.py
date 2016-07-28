import pickle, os
import xgboost as xgb
from matplotlib import pyplot as plt

def print_importances(model, features):
	booster = model._Booster
	booster.feature_names = [name for (name, fn) in features]
	scores = list(booster.get_fscore().iteritems())
	ranked = sorted(scores, key=lambda (k,v): -v)
	print "feature importances:"
	for t in ranked: print "\t", t

def visualize():
	from ml import feature_defs
	features = feature_defs()
	print "loading model..."
	with open("pickle/model.pickle", 'r') as f:
		model = pickle.load(f)

	print_importances(model, features)

	# print "feature importances:"
	# print booster.get_fscore()

	# for i, (name, fn) in enumerate(features):
	#  	print name, model.feature_importances_[i]

	print "plot importance:"
	xgb.plot_importance(model).plot()
	plt.show()

if __name__ == '__main__':
	visualize()

