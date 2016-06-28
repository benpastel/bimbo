import numpy as np
import pandas as pd
from random import random

train_file = "data/joined.csv"
train_cols = (
	"client_id",
	"product_id",
	"week",
	"sales_depot_id",
	"sales_channel_id",
	"route_id",
	"units_sold",
	"value_sold",
	"units_returned",
	"value_returned",
	"net_units_sold",
	"client_name",
	"product_name"
)

# TODO: use only weeks that we want at a given time
# TODO: also try sampling by client, or by product

total_count = 74180464
target_sample_size = 500000
sample_rate = float(target_sample_size) / float(total_count)

print "sampling from training file..."
lines = []
with open(train_file, 'r') as f:
	for line in f:
		if random() < sample_rate:
			tokens = line.strip().split(",")
			if len(tokens) != 13:
				print "malformed line:", line
			else:	
				lines.append(tokens)
print "loaded %d samples" % len(lines)

train = pd.DataFrame.from_records(lines, columns=(train_cols), 
	index=("client_id","product_id"))



# parse the data with pandas / numpy?



