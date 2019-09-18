#!/bin/bash
################################
if [ -z "$1" ]
then
echo "$0 reservation        (from local) reserve nodes and store job ID"
echo "$0 run                (from the job on g5k) run simulations"
echo "$0 download           (from local) download log files"
exit 0
fi
##################################
if [ $1 = "reservation" ]
then
  if [ -z $2 ] || [ -z $3 ]
  then
    echo "$0 $1 NB_HOSTS NB_HOURS"
  else
    ssh nancy.g5k "oarsub -q production -p \"cluster='graffiti'\" -l nodes=$2,gpu=1,walltime=$3  \"sleep 10d\""
    job=$(ssh nancy.g5k "oarstat | grep tanne")
    ssh nancy.g5k 
  fi
##################################
#################################
################################
elif [ $1 = "run" ]
then
  if [ -z $2 ] || [ -z $3 ]
  then
    echo "$0 $1 Conda_Env Python_File"
    exit 0
  else
    nodes=$(uniq $OAR_NODEFILE)
    > /home/tanne/finished_worker.txt
    shift
    for n in $nodes
    do
      oarsh tanne@$n "./Documents/Prelab_LARSEN/handful-of-trials/scripts/python_script.sh $@" &
    done
  fi 
fi
  
