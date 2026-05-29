#!/bin/bash
# ============================================================================
# Evaluate all 5 models on multilingual & child-speech test sets
# Produces per-language WER comparison table
# ============================================================================

set -euo pipefail

PROJECT_DIR=/lp-dev/amelia/inclusive-asr-moe
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/multilingual}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multilingual & Child-Speech Evaluation"
echo "=========================================="
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo ""

# Define models
declare -A MODELS=(
    ["dense_adult"]="/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual/dense/dense_fastconformer_multilingual_2026-04-22_13-48-35/2026-04-22_13-48-44/checkpoints/dense_fastconformer_multilingual_2026-04-22_13-48-35.nemo"
    ["moe_adult"]="/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual/moe/moe_fastconformer_multilingual_2026-04-22_13-48-13/2026-04-22_13-48-26/checkpoints/moe_fastconformer_multilingual_2026-04-22_13-48-13.nemo"
    ["dense_child"]="/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual_child/dense_fastconformer_child_myst_2026-04-30_10-32-43/2026-04-30_10-32-53/checkpoints/dense_fastconformer_child_myst_2026-04-30_10-32-43.nemo"
    ["moe_child_lb_off"]="/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual_child/moe/moe_fastconformer_child_multilingual_load_balancing_on_2026-05-04_09-56-09/2026-05-04_09-56-22/checkpoints/moe_fastconformer_child_multilingual_load_balancing_on_2026-05-04_09-56-09.nemo"
    ["moe_child_lb_on"]="/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual_child/moe/moe_fastconformer_child_multilingual_2026-05-05_15-18-57/2026-05-05_15-19-10/checkpoints/moe_fastconformer_child_multilingual_2026-05-05_15-18-57.nemo"
)

# Test sets
ADULT_LANGUAGES="en_librispeech nl de pl"
CHILD_LANGUAGES="child_en_myst child_nl_jasmin child_en_kidstalc child_pl_pavsig"

# Evaluate adult-trained models on all test sets
echo "=== ADULT-TRAINED MODELS (evaluated on all test sets) ==="
echo ""

for model_key in "dense_adult" "moe_adult"; do
    checkpoint="${MODELS[$model_key]}"
    model_name=$(echo "$model_key" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
    
    echo "Evaluating: $model_name"
    OUTPUT_JSON="$OUTPUT_DIR/${model_key}_results.json"
    
    python3 ${PROJECT_DIR}/evaluation/evaluate_multilingual.py \
        --checkpoint "$checkpoint" \
        --languages $ADULT_LANGUAGES $CHILD_LANGUAGES \
        --output "$OUTPUT_JSON" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
    
    echo "  Saved: $OUTPUT_JSON"
    echo ""
done

# Evaluate child-finetuned models on all test sets
echo "=== CHILD-FINETUNED MODELS (evaluated on all test sets) ==="
echo ""

for model_key in "dense_child" "moe_child_lb_off" "moe_child_lb_on"; do
    checkpoint="${MODELS[$model_key]}"
    model_name=$(echo "$model_key" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
    
    echo "Evaluating: $model_name"
    OUTPUT_JSON="$OUTPUT_DIR/${model_key}_results.json"
    
    python3 ${PROJECT_DIR}/evaluation/evaluate_multilingual.py \
        --checkpoint "$checkpoint" \
        --languages $ADULT_LANGUAGES $CHILD_LANGUAGES \
        --output "$OUTPUT_JSON" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS"
    
    echo "  Saved: $OUTPUT_JSON"
    echo ""
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results in: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "To process results into a table, run:"
echo "  python3 ${PROJECT_DIR}/evaluation/compile_results.py \\""
echo "    --results_dir $OUTPUT_DIR"
echo ""
