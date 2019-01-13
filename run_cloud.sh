#!/bin/bash 
gcloud ml-engine jobs submit training learn_map_15 \
    --job-dir=gs://learn-map-227816-mlengine/learn_map_15 \
    --module-name trainer.task \
    --packages=dist/learn_map-0.1.tar.gz \
    --python-version 3.5 \
    --region us-central1 \
    --config=cloudml-gpu.yaml \
    --runtime-version 1.12 \
    --  \
    --num-epochs=5 \
    --eval-batch-size=1000 \
    --train-batch-size=4096 \
    --train-steps=200 \
    --grid-size=22 \
    --train
