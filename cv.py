import numpy as np
import pandas as pd
import os
from models import *

train_pickle = "pickle/slim_train_df.pickle"
dev_pickle = "pickle/slim_dev_df.pickle"
train_cols = (
	"week",
	"client_id",
	"product_id",
	"net_units_sold"
)
train_dtypes = {
	"week": np.int8,
	"client_id": np.int32,
	"product_id": np.int32,
	"net_units_sold": np.int32
}
train_weeks = range(3, 9)

model_fns = [
	int_avg, 
	simple_avg,
	log_avg
]

def RMSLE(preds, actuals):
	diffs = np.log(preds + 1) - np.log(actuals + 1)
	return np.sqrt( np.average(diffs ** 2) )

if not os.path.isfile(train_pickle) or not os.path.isfile(dev_pickle):
	print "loading training data from csv..."
	weekly_data = {}
	for week in range(3, 10):
		with open("split/train_%d.csv" % week, 'r') as f:
			data = pd.read_csv(f, names=train_cols, dtype=train_dtypes, engine='c')
			data["key"] = data["client_id"].astype(np.int64) * 50000 + data["product_id"] # product_ids are < 50k
			weekly_data[week] = data
			print "week %d: %d lines" % (week, len(data))

	dev = weekly_data[9]
	train = pd.concat([weekly_data[w] for w in range(3, 9)])

	print "saving %d train lines, %d dev lines to pickle" % (len(train), len(dev))
	train.to_pickle(train_pickle)
	dev.to_pickle(dev_pickle)

else:
	print "loading pickles..."
	train = pd.read_pickle(train_pickle)
	dev = pd.read_pickle(dev_pickle)

dev = dev.sample(n = 1000000)
print "using %d train, %d dev" % (len(train), len(dev))

for model_fn in model_fns:
	print "making predictions with " + str(model_fn) + "..."
	preds = model_fn(train, dev)
	print "RMSLE: ", RMSLE(preds, dev["net_units_sold"])

