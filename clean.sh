#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

compare_line_count() {
	count1=`wc -l $1 | cut -d' ' -f3`
	count2=`wc -l $2 | cut -d' ' -f3`
	echo $(( count1 - count2 ))
}

if [ ! -f data/clients.csv ]; then
	echo "Cleaning clients..."

	# drop the header
	sed '1d' raw_data/cliente_tabla.csv > tmp/clients.csv	

	# collapse whitespace and dedupe
	sed -E 's/[[:space:]]+/ /g' tmp/clients.csv | sort -u > tmp/clients2.csv
	echo -n "clients deduped by whitespace: " 
	compare_line_count 'tmp/clients.csv' 'tmp/clients2.csv'

	# remove the one remaining duplicate manually
	grep -v '1646352,SIN NOMBRE' tmp/clients2.csv > tmp/clients3.csv
	echo -n "clients deduped manually: " 
	compare_line_count 'tmp/clients2.csv' 'tmp/clients3.csv'
	mv tmp/clients3.csv data/clients.csv
fi

if [ ! -f data/products.csv ]; then
	echo "Cleaning products..."

	sed '1d' raw_data/producto_tabla.csv > tmp/products.csv
	mv tmp/products.csv data/products.csv
fi

if [ ! -f data/train.csv ]; then 
	echo "Cleaning training data..."

	sed '1d' raw_data/train.csv > tmp/train.csv
	mv tmp/train.csv data/train.csv	
fi

if [ `sqlite3 bimbo.db "select count(*) FROM sqlite_master WHERE type='table' AND name='train';"` -eq 0 ] ; then
	echo "creating table..."
	sqlite3 -echo bimbo.db < create.sql
fi

if [ ! -f data/joined.csv ]; then
	echo "Joining data..."
	sort -b -t, -k1 data/clients.csv > tmp/c
	sort -b -t, -k5 data/train.csv > tmp/t
	join -t, -1 5 -2 1 tmp/t tmp/c > tmp/joined

	sort -b -t, -k1 data/products.csv > tmp/p
	sort -b -t, -k6 tmp/joined > tmp/t2
	join -t, -1 6 -2 1 tmp/t2 tmp/p > data/joined.csv
fi

if [ ! -d split ]; then
	echo "Preparing to split by week"
	mkdir split/
	sort -g -t, -k3 data/joined.csv > split/all
fi


# if [ ! -d split ]; then
# 	echo "Splitting data by week..."
# 	mkdir split/
# 	cp data/train.csv split/all

# 	for w in `seq 3 8`; do 
# 	

# next=$((w + 1))
# dst="split/train_$w.csv"

# 	# find the first line with the next week
# 		line=`cut -d, -f3 split/all | nl | grep $next'$' | head -1 | cut -f1`
# 		count=$((line - 1))

# 		head -$count split/all > $dst
# 		sed "${count}d" split/all > split/tmp
# 		mv split/tmp split/all

# 	
# 	done
# 	mv split/all split/train_9.csv
# fi
