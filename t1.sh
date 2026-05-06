#!/bin/bash

python3 ./tools/plot_sim.py --binary ./build/dd_mpc/sim_test --random --complexity 5 --stages 2 --seed 4 --config ./config/mpc_default.cfg
