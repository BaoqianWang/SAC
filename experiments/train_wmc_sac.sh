#!/bin/sh
python3 sac_benchmark.py \
    --max-iteration=3000 \
    --evalM=15  \
    --save-dir="../result/wmc/sac/" \
    --save-rate=200 \
    --T=25 \
    --grid-size=3 \
    --ddl=10 \
    --gamma=0.95 \
    --seed=25 \
