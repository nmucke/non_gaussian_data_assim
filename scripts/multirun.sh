#!/bin/bash

for infl in $(seq 1.0 0.05 2.0)
#for rloc in $(seq 10 10 300)
do
	echo "Inflation factor: $infl"
	#echo "Localisation radius: $rloc"
	uv run main.py case=kuramoto da_method=enkf inflation_factor=$infl
	#uv run main.py case=kuramoto da_method=enkf localization_distance=$rloc
done
