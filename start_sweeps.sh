#!/bin/bash

# Start four sweeps
for i in {0..3}; do
    if [ $((i % 2)) -eq 0 ]; then
        gpu_number="4"
    else
        gpu_number="5"
    fi
    nohup python3 kfold_hypersweep.py --gpu_number $gpu_number > "sweep$i.out" &
done

# Wait for all sweeps to finish
wait