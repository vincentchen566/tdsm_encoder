#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh
python3 -V
echo "$(ls)"
echo "$(nvidia-smi -L)"
echo $CUDA_VISIBLE_DEVICES
CFG_FILE=$(echo ${1} | cut -d '/' -f 3)
echo $CFG_FILE
python3 trans_tdsm.py -s 0010 -i ds2_diff_transforms -c $CFG_FILE