#!/bin/sh
# pass the ssh public key of host to ec2 instances
filename='nodeIPaddress'
while read line; do
echo $line
# scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/darl1n_neurlps/amazon_scripts/
scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ubuntu@3.145.25.30:/home/ubuntu/SAC/result/wmc/sac/9_agents_2_ddl_30000_iteration_16_seed.zip  ~/sac/result/wmc/sac/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ubuntu@3.144.5.124:/home/ubuntu/darl1n_neurlps/result/grassland/darl1n_ec2/48agents_fixed_seed16/good_agent.pkl  ~/darl1n_neulps/result/grassland/darl1n/48agents_fixed_2_seed16/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ubuntu@3.144.5.124:/home/ubuntu/darl1n_neurlps/result/grassland/darl1n_ec2/48agents_fixed_seed16/global_time.pkl  ~/darl1n_neulps/result/grassland/darl1n/48agents_fixed_2_seed16/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ubuntu@3.144.5.124:/home/ubuntu/darl1n_neurlps/result/grassland/darl1n_ec2/48agents_fixed_seed16/train_time.pkl  ~/darl1n_neulps/result/grassland/darl1n/48agents_fixed_2_seed16/


# scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/darl1n_neurlps/maddpg_o/experiments
echo "Transfer $1 to $line Done !"
done < $filename
