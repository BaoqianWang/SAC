#!/bin/sh
mpirun -n 10 python3 train_darl1n.py \
    --scenario=wireless_mc \
    --num-agents=9 \
    --num-learners=9  \
    --save-dir="../result/wmc/darl1n/9agents/" \
    --save-rate=10 \
    --max-num-train=200 \
    --max-num-neighbors=5 \
    --eva-max-episode-len=25 \
    --seed=19 \
