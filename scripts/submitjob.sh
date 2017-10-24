#! /bin/bash

export LOCALDIR="/home/obs006/src/IMLE/imle"
echo "Starting submission process...${LOCALDIR}"

for i in {1..5}; do
    for j in {1..9}; do
	cd ${LOCALDIR}/scripts
	export run=$i
	export example=$j
	sbatch --export=ALL ./run.sbatch.sh
	sleep 1
    done
done
