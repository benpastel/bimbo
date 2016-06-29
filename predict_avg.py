from __future__ import print_function
import numpy as np

test_file = "data/test.csv"
train_file = "data/slim_train.csv"
median_sales = 3

test_ids = {} # (client_id, product_id) => {test_ids}
sales_logsum = {} # key => sum of log(sales) in training set
sales_count = {} # key => count of sales in training set

print("loading test set...")
count = 0;
with open(test_file, 'r') as f:
	for line in f:
		count += 1
		test_id, _, _, _, _, client_id, product_id = line.strip().split(',')
		pair_key = client_id + "_" + product_id
		client_key = "c_" + client_id 
		product_key = "p_" + product_id

		ids = test_ids.get(pair_key, set())
		ids.add(test_id)
		test_ids[pair_key] = ids

		sales_logsum[pair_key] = 0
		sales_logsum[client_key] = 0
		sales_logsum[product_key] = 0
		sales_count[pair_key] = 0
		sales_count[client_key] = 0
		sales_count[product_key] = 0
print("loaded %d (client, product) pairs in %d lines" % (len(test_ids), count))

print("scanning train set...")
count = 0;
useful_pairs = 0;
useful_clients = 0;
useful_products = 0;
def increment(key, sales):
	if key not in sales_logsum:
		return False
	sales_logsum[key] += np.log(sales)
	sales_count[key] += 1
	return True

with open(train_file, 'r') as f:
	for line in f:
		count += 1
		_, client_id, product_id, sales = line.strip().split(',')
		sales = int(sales)
		pair_key = client_id + "_" + product_id
		client_key = "c_" + client_id
		product_key = "p_" + product_id

		pair_used = increment(pair_key, sales)
		client_used = increment(client_key, sales)
		product_used = increment(product_key, sales)

		if pair_used: useful_pairs += 1
		if client_used: useful_clients += 1
		if product_used: useful_products += 1
print("used %d pairs, %d clients, %d products, out of %d total lines" % (useful_pairs, useful_clients, useful_products, count))

print("printing predictions...")
pair_preds = 0
combined_preds = 0
median_preds = 0
with open("pred/log_avgs.csv", 'w') as f:
	for pair_key in test_ids.keys():
		client_id, product_id = pair_key.split("_")
		client_key = "c_" + client_id
		product_key = "p_" + product_id

		if sales_count[pair_key] > 0:
			pred = np.exp(sales_logsum[pair_key]) / sales_count[pair_key]
			pair_preds += 1
		# elif sales_count[client_key] > 0 and sales_count[product_key] > 0:
		# 	pred = (sales_sum[client_key] + sales_sum[product_key]) / (sales_count[client_key] + sales_count[product_key])
		# 	combined_preds += 1
		else:
			pred = median_sales
			median_preds += 1
		for test_id in test_ids[pair_key]:
			print("%s,%d" % (test_id, pred), file=f)
print("Prediction count: %d by pair, %d by client & product avg, %d by median" 
	% (pair_preds, combined_preds, median_preds))
print("Done.")




