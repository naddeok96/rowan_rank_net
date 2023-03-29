#!/bin/bash

# Activate the virtual environment
source ../envs/rowanranknet/bin/activate

# Start four sweeps
for i in {0..3}; do
    # Determine the GPU number to use for the current run based on the loop index
    if [ $((i % 2)) -eq 0 ]; then
        gpu_number="4"
    else
        gpu_number="5"
    fi

    # Run the kfold_hypersweep.py script with the specified GPU number in the background
    # Redirect the output of the script to a file named sweep$i.out
    nohup python3 kfold_hypersweep.py --gpu_number $gpu_number > "sweep$i.out" &
done

# Wait for all sweeps to finish before exiting
wait
