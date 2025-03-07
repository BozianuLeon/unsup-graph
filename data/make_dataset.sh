#!/bin/bash

root="/srv/beegfs/scratch/shares/atlas_caloM/mu_200_truthjets/"
k=3
r=150
output_dir="/home/users/b/bozianu/work/calo-cluster/unsup-graph/"

#knn
# cmd="python /home/users/b/bozianu/work/calo-cluster/unsup-graph/data/dataset.py --root $root --name knn -k $k -o $output_dir"
#rad
# cmd="python /home/users/b/bozianu/work/calo-cluster/unsup-graph/data/dataset.py --root $root --name rad -r $r -o $output_dir"
#bucket
cmd="python /home/users/b/bozianu/work/calo-cluster/unsup-graph/data/dataset.py --root $root --name bucket -o $output_dir"
echo $cmd
eval $cmd 