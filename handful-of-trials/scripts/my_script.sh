#!/bin/sh
BIN=$2
export PATH=/home/tanne/miniconda3/bin:$PATH
. /home/tanne/miniconda3/etc/profile.d/conda.sh
conda activate $1
shift
shift 
exec python $BIN $@
