#!/usr/bin/ksh
ARRAY=()
SchemeArray=()
host_name=""
k=0
l=0

while read LINE
do
    ARRAY+=("$LINE")
    ((k=k+1))
done < nodeIPaddress

sleep 1
echo "Wireless Multi Access"

host_name_uncoded="${ARRAY[1]}"
for((i=2;i<=10;i++))
do
  host_name_uncoded="${host_name_uncoded},${ARRAY[i]}"
done

mpirun --mca plm_rsh_no_tree_spawn 1 --mca btl_base_warn_component_unused 0  --host $host_name_uncoded \
python3 ../experiments/train_darl1n.py \
    --scenario=wireless_mc \
    --num-agents=9 \
    --num-learners=9 \
    --save-dir="../result/wmc/darl1n/" \
    --save-rate=10 \
    --max-num-train=400 \
    --max-num-neighbors=9 \
    --eva-max-episode-len=25 \
    --ddl=10 \
    --seed=16 \
