#!/bin/bash

configs=(
    "power"
    "intermediate"
    "precision"
    "all"
)

for config in "${configs[@]}" ; do
    echo "${config}"

    # eigen_nerf_w_pe
    python run_eigen_nerf.py --config configs/multiple_behaviors/exp_eigen_nerf_w_pe_1000/${config}.txt
done