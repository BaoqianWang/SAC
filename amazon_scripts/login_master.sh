#!/usr/bin/ksh
ARRAY=()
while read LINE
do
    ARRAY+=("$LINE")
done < nodeIPaddress
echo ${ARRAY[$1]}
ssh  -i ~/AmazonEC2/.ssh/linux_key_pari.pem ubuntu@${ARRAY[$1]}
