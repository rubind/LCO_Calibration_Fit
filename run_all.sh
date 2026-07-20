#!/usr/bin/env bash

set -e

runs=(
  "U U"
  "U_2 U"
  "U_3 U"
  "B B"
  "B_2 B"
  "V V"
  "V_2 V"
  "R R"
  "R_2 R"
  "I I"
  "I_2 I"
  "up up"
  "up_2 up"
  "up_3 up"
  "gp gp"
  "gp_2 gp"
  "rp rp"
  "rp_2 rp"
  "ip ip"
  "ip_2 ip"
  "zs zs"
  "zs_2 zs"
)

for run in "${runs[@]}"; do
    read -r config_name filter_name <<< "$run"

    PYTHONPATH=src /opt/anaconda3/bin/python -m passband_fit fit \
        --config "configs/${config_name}.yaml" \
        --filter "$filter_name" \
        --run-label "${config_name}-production-seed12345"
done


(  "U U"
  "U_2 U"
  "U_3 U"
  "B B"
  "B_2 B"
  "V V"
  "V_2 V"
  "R R"
  "R_2 R"
  "I I"
  "I_2 I"
  "up up"
  "up_2 up"
  "up_3 up"
  "gp gp"
  "gp_2 gp"
  "rp rp"
  "rp_2 rp"
  "ip ip"
  "ip_2 ip"
  "zs zs"
  "zs_2 zs"
)

