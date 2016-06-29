import numpy as np
import pandas as pd
from random import random

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
train_weeks = range(3, 8)
dev_weeks = (8, 9)

total_count = 74180464
# target_sample_size = 500000
# sample_rate = float(target_sample_size) / float(total_count)

print "loading training file..."
all_train = pd.read_csv(train_file, names=train_cols, dtype=train_dtypes, engine='c')
print "loaded %d samples" % len(train)

print "splitting into train / dev..."
train = all_train[all_train['week'].isin(train_weeks)]
dev = all_train[all_train['week'].isin(dev_weeks)]
print "%d in train, %d in dev" % (train, dev)


def RMSLE(preds, actuals):
	diffs = np.log(preds + 1) - np.log(actuals + 1)
	return np.sqrt( np.average(diffs ** 2) )
