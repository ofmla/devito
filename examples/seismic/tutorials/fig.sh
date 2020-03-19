#!/bin/bash

#rm *.pdf


file1=${1}.file
file2=${1}.ps

echo $file1
/home/oscar/SU/bin/psimage n1=5001 n2=42 < ${file1} > ${file2} perc=99 d1=0.004. f1=0. d1num=5.0 f1num=0.0 d2=1. f2=1. d2num=10.0 f2num=10.0 width=4.0 height=6.0 axeswidth=1.1 bps=24 label1='Time (sec)' label2='Traces' labelsize=18

ps2eps -B -r 600 ${file2}
epstopdf  ${1}.eps

#exit 0
