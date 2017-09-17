#!/bin/bash
methods=(old_r_BN old_r_WN_BN old_r_CWN_BN_NS)
lrs=(0.1)

batchSize=128
weightDecay=0.0001
dr=0


n=${#methods[@]}
m=${#lrs[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   CUDA_VISIBLE_DEVICES=0	th exp_GoogleNet_dataWhitening.lua -model ${methods[$i]} -learningRate ${lrs[$j]}  -max_epoch 160 -seed 1 -dropout ${dr}  -batchSize ${batchSize} -weightDecay ${weightDecay} -depth 56
   done
done
