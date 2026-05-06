#!/bin/bash

python3 ./tools/plot_sim.py --binary ./build/dd_mpc/sim_test --random --complexity 5 --stages 20 --seed 4 --config ./config/mpc_default.cfg
