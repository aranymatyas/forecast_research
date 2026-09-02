#!/bin/bash
# Run all 9 dataset-model combinations by executing the pipeline notebook.
# All settings come from the config JSON (set via the CONFIG env var; defaults
# to config.json); this script only swaps the dataset name and model per run.
#
# Usage: nohup ./run_all.sh > run_all.log 2>&1 &
#        CONFIG=my_config.json ./run_all.sh

set -e

CONFIG="${CONFIG:-config.json}"
NOTEBOOK="${NOTEBOOK:-nb/m_datasets/pipeline.ipynb}"

DATASETS=("web-traffic" "M5" "M4")
MODELS=("AutoETS" "AutoARIMA" "Prophet")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "$(date): Starting $dataset / $model"

        tmp="$(mktemp)"
        jq --arg dataset "$dataset" --arg model "$model" \
            '.dataset.name = $dataset | .model = $model' \
            "$CONFIG" > "$tmp" && mv "$tmp" "$CONFIG"

        CONFIG="$CONFIG" jupyter nbconvert \
            --to notebook \
            --execute \
            --inplace \
            "$NOTEBOOK"

        echo "$(date): Finished $dataset / $model"
        echo ""
    done
done

echo "$(date): All runs complete."
