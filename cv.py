import numpy as np
import pandas as pd
from models import *

train_file = "data/slim_train.csv"
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
dev_weeks = range(9, 10)

model_fns = [
	int_avg, 
	simple_avg,
	log_avg
]

def RMSLE(preds, actuals):
	diffs = np.log(preds + 1) - np.log(actuals + 1)
	return np.sqrt( np.average(diffs ** 2) )

print "loading training file..."
all_train = pd.read_csv(train_file, names=train_cols, dtype=train_dtypes, engine='c')

# the product_ids are all less than 50k
all_train["key"] = all_train["client_id"].astype(np.int64) * 50000 + all_train["product_id"]
print "loaded %d samples" % len(all_train)

print "splitting into train / dev..."
train = all_train[all_train['week'].isin(train_weeks)]
dev = all_train[all_train['week'].isin(dev_weeks)]
dev = dev.sample(n = 1000000) # downsample
print "%d in train, %d in dev, %d total" % (len(train), len(dev), len(all_train))

for model_fn in model_fns:
	print "making predictions with " + str(model_fn) + "..."
	preds = model_fn(train, dev)
	print "RMSLE: ", RMSLE(preds, dev["net_units_sold"])

