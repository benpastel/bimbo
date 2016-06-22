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

if [[ $( sqlite3 bimbo.db "SELECT 1 - count(*) FROM sqlite_master WHERE type='table' AND name='train';" ) ]] ; then
	echo "creating table..."
	sqlite3 -echo bimbo.db < create.sql
fi
