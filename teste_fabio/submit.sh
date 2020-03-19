#!/bin/bash

nrun=$1
export DEVITO_OPENMP=1
ulimit -s unlimited
mprof run ./utest_nodask_naws.py > ${nrun}.txt

