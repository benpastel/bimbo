#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

echo "finding unique (client, product) pairs..."
cut -d, -f6 -f7 data/test.csv | sort -b -u > tmp/test_pairs
cut -d, -f5 -f6 data/train.csv | sort -b -u > tmp/train_pairs

echo "comparing file sizes:"
join tmp/test_pairs tmp/train_pairs > tmp/both_pairs
wc -l 'data/test.csv tmp/test_pairs data/train.csv tmp/train_pairs tmp/both_pairs'