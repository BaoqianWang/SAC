#!/bin/sh
mpirun -n 10 python3 train_darl1n.py \
    --scenario=wireless_mc \
    --num-agents=9 \
    --num-learners=9  \
    --save-dir="../result/wmc/darl1n/" \
    --save-rate=10 \
    --max-num-train=100 \
    --max-num-neighbors=9 \
    --eva-max-episode-len=25 \
    --ddl=10\
    --seed=25 \
