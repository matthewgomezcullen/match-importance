#!/bin/bash

for year in {17..23}
do
    python run.py "${year}_$((year+1))" &
done

wait

python merge.py

echo "All seasons have been processed."