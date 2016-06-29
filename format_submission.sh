#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

if [ "$#" -ne 1 ]; then
    echo -e "\nUsage:\n$0 [unordered csv of (id , prediction)] \n"
fi

echo "id,Demanda_uni_equil" > pred/header.txt
sort -g -t, -k1 $1 > tmp/sorted_pred.csv
cat pred/header.txt tmp/sorted_pred.csv > pred/submission.csv
