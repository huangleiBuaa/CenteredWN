#!/bin/bash
methods=(sgd nnn CWN_Row_scale WN_Row_scale)

lrs=(0.001 0.002 0.005 0.011)
nls=(2)

n=${#methods[@]}
m=${#lrs[@]}
f=${#nls[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "nonlinear=${nls[$j]}"
   	th exp_MLP.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -mode_nonlinear ${nls[$k]} -optimization simple -seed 1 -batchSize 1024
      done
   done
done
