#!/bin/bash 
gcloud ml-engine jobs submit training learn_map_17 \
    --job-dir=gs://learn-map-227816-mlengine/learn_map_17 \
    --module-name trainer.task \
    --packages=dist/learn_map-0.1.tar.gz \
    --python-version 3.5 \
    --region us-central1 \
    --runtime-version 1.12 \
    --scale-tier STANDARD_1 \
    --  \
    --num-epochs=5 \
    --eval-batch-size=1000 \
    --train-batch-size=4096 \
    --train-steps=150 \
    --grid-size=22 \
    --train=True \
    --evaluate=True
