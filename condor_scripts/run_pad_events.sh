#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh
python3 -V
INFILE=$1
echo "$(ls)"
echo "$(nvidia-smi -L)"
echo $CUDA_VISIBLE_DEVICES
python3 pad_events.py -i /eos/user/t/tihsu/database/ML_hackthon/bucketed_tensor/ -f $INFILE -o ds2_newpp -t 1