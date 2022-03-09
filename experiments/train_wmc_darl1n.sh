#!/bin/sh
mpirun -n 10 python3 train_darl1n.py \
    --scenario=wireless_mc \
    --num-agents=9 \
    --num-learners=9  \
    --save-dir="../result/wmc/darl1n/9agents_2000/" \
    --save-rate=10 \
    --max-num-train=150 \
    --max-num-neighbors=9 \
    --eva-max-episode-len=25 \
    --seed=25 \
