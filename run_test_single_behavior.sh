#!/bin/bash

configs=(
    "ball_tennis_power_sphere"
    "water_bottle_medium_wrap"
    "peco_can"
    "tooth_brush"
    "cake_pan_square"
    "chip_casino_lateral"
    "pole"
    "chip_casino_palmar_pinch"
    "ball_tennis_tripod"
    "pin_hair"
    "dropping_lid"
)

for config in "${configs[@]}" ; do
    echo "${config}"

    # eigen_nerf_w_pe
    python run_eigen_nerf_eval.py --mode single_action --config configs/single_behavior/exp_eigen_nerf_w_pe_1000/${config}.txt
done