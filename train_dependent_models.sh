#!/bin/bash

# Loop through participants 4 to 32
for participant in {5..32}; do
    echo "Training model for participant $participant..."
    python subject_dependent.py --participant "$participant"
done

echo "All participants processed."