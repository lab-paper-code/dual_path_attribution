#!/bin/bash

# Configuration
export CUDA_VISIBLE_DEVICES=2
model="meta-llama/Llama-3.1-8B-Instruct"  
input_val=${1:-1.0}         

format_val() {
    local val="$1"
    # If the value has exactly two decimal places and ends in 0 (e.g., 0.50, 0.00), 
    # capture everything up to the first decimal digit and drop the trailing zero.
    if [[ "$val" =~ ^([0-9]+\.[0-9])0$ ]]; then
        val="${BASH_REMATCH[1]}"
    fi
    echo "${val//./}"
}

c_025="025"
c_05="05"

for p in  0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    res=$(echo "scale=2; $input_val - $p" | bc)
    res=$(printf "%.1f" "$res")
    p_half=$(printf "%.2f" "$(echo "scale=2; $p / 2" | bc)")
    res_half=$(printf "%.2f" "$(echo "scale=2; $res / 2" | bc)")

    c_p=$(format_val "$p")                 # 0.0 -> 00, 1.0 -> 10
    c_res=$(format_val "$res")             # 1.0 -> 10, 0.5 -> 05
    c_res_half=$(format_val "$res_half")   # 0.50 -> 05, 0.25 -> 025, 0.00 -> 00
    c_p_half=$(format_val "$p_half")

    echo "--- Running experiments for p=$p (res=$res) ---"

    # Input Attribution
    python3 -m experiments.attribution.attribution_pipe \
        --attribution input --dataset known_1000 --model "$model" --batch_size 64 \
        --methods dpa --dpa_weights "$res_half" "$res_half" "$p" "$res" "$p"

    python3 -m experiments.attribution.attribution_pipe \
        --attribution input --dataset known_1000 --model "$model" --batch_size 64 \
        --methods dpa --dpa_weights "$res_half" "$p_half" 0.5 0.5 0.5
    
    python3 -m experiments.attribution.attribution_pipe \
        --attribution input --dataset known_1000 --model "$model" --batch_size 64 \
        --methods dpa --dpa_weights "$res_half" "$res_half" "$p" 0.5 0.5
    
    python3 -m experiments.attribution.attribution_pipe \
        --attribution input --dataset known_1000 --model "$model" --batch_size 64 \
        --methods dpa --dpa_weights 0.25 0.25 0.5 "$res" "$p"

    # # Ablation Pipe (Disrupt) - Uses formatted strings
    python3 -m experiments.ablation.ablation_pipe \
        --ablation input --type disrupt --dataset known_1000 --model "$model" --batch_size 1024 \
        --method "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_res}_${c_p}" \
                 "dpa_${c_res_half}_${c_p_half}_${c_05}_${c_05}_${c_05}" \
                 "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_05}_${c_05}" \
                 "dpa_${c_025}_${c_025}_${c_05}_${c_res}_${c_p}"

    # # Ablation Pipe (Recover) - Uses formatted strings
    python3 -m experiments.ablation.ablation_pipe \
        --ablation input --type recover --dataset known_1000 --model "$model" --batch_size 1024 \
        --method "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_res}_${c_p}" \
                 "dpa_${c_res_half}_${c_p_half}_${c_05}_${c_05}_${c_05}" \
                 "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_05}_${c_05}" \
                 "dpa_${c_025}_${c_025}_${c_05}_${c_res}_${c_p}"
done

for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

    res=$(echo "scale=2; $input_val - $p" | bc)
    res=$(printf "%.1f" "$res")
    p_half=$(printf "%.2f" "$(echo "scale=2; $p / 2" | bc)")
    res_half=$(printf "%.2f" "$(echo "scale=2; $res / 2" | bc)")

    c_p=$(format_val "$p")                 # 0.0 -> 00, 1.0 -> 10
    c_res=$(format_val "$res")             # 1.0 -> 10, 0.5 -> 05
    c_res_half=$(format_val "$res_half")   # 0.50 -> 05, 0.25 -> 025, 0.00 -> 00
    c_p_half=$(format_val "$p_half")

    # Component Attribution
    python3 -m experiments.attribution.attribution_pipe \
        --attribution component --dataset known_1000 --model "$model" --batch_size 4 \
        --methods dpa --dpa_weights "$res_half" "$res_half" "$p" "$res" "$p"

    python3 -m experiments.attribution.attribution_pipe \
        --attribution component --dataset known_1000 --model "$model" --batch_size 4 \
        --methods dpa --dpa_weights "$res_half" "$p_half" 0.5 0.5 0.5
    
    python3 -m experiments.attribution.attribution_pipe \
        --attribution component --dataset known_1000 --model "$model" --batch_size 4 \
        --methods dpa --dpa_weights "$res_half" "$res_half" "$p" 0.5 0.5
    
    python3 -m experiments.attribution.attribution_pipe \
        --attribution component --dataset known_1000 --model "$model" --batch_size 4 \
        --methods dpa --dpa_weights 0.25 0.25 0.5 "$res" "$p"

    # # Ablation Pipe (Disrupt) - Uses formatted strings
    python3 -m experiments.ablation.ablation_pipe \
        --ablation component --type disrupt --dataset known_1000 --model "$model" --batch_size 16 \
        --method "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_res}_${c_p}" \
                 "dpa_${c_res_half}_${c_p_half}_${c_05}_${c_05}_${c_05}" \
                 "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_05}_${c_05}" \
                 "dpa_${c_025}_${c_025}_${c_05}_${c_res}_${c_p}"

    # # Ablation Pipe (Recover) - Uses formatted strings
    python3 -m experiments.ablation.ablation_pipe \
        --ablation component --type recover --dataset known_1000 --model "$model" --batch_size 16 \
        --method "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_res}_${c_p}" \
                 "dpa_${c_res_half}_${c_p_half}_${c_05}_${c_05}_${c_05}" \
                 "dpa_${c_res_half}_${c_res_half}_${c_p}_${c_05}_${c_05}" \
                 "dpa_${c_025}_${c_025}_${c_05}_${c_res}_${c_p}"
done