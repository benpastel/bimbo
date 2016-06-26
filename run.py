import numpy as np
import pandas as pd
from random import random

train_file = "data/joined.csv"

# load a sample of the data
# TODO: use only weeks that we want at a given time
# TODO: also try sampling by client, or by product
total_count = 74180464
target_sample_size = 500000
sample_rate = float(target_sample_size) / float(total_count)

print "sampling from training file..."
train_lines = []
with open(train_file, 'r') as f:
	for line in f:
		if random() < sample_rate:
			train_lines.append(line)
print "loaded %d samples" % len(train_lines)