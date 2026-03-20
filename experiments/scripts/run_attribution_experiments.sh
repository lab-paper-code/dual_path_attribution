#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for model in meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-2-7b-chat-hf Qwen/Qwen3-4B-Instruct-2507 mistralai/Mistral-7B-Instruct-v0.3; do

    for dataset in known_1000 squad_v2.0 imdb; do
        for method in random last_layer_attn mean_attn attn_rollout gradient input_x_gradient integrated_gradient depass ifr ap; do
            python3 -m experiments.attribution.attribution_pipe \
                --attribution input \
                --dataset "${dataset}" \
                --model $model \
                --batch_size 4 \
                --chunk_size 4 \
                --methods $method
        done

        python3 -m experiments.ablation.ablation_pipe \
            --ablation component \
            --type disrupt \
            --dataset "${dataset}"  \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --batch_size 16

        python3 -m experiments.ablation.ablation_pipe \
            --ablation component \
            --type recover \
            --dataset "${dataset}"  \
            --model meta-llama/Llama-3.1-8B-Instruct \
            --batch_size 16
    done

    for dataset in known_1000 ioi; do
        for method in random attn_act mlp_act norm gradient atp ifr dpa norm ap; do
            python3 -m experiments.attribution.attribution_pipe \
                --attribution component \
                --dataset "${dataset}" \
                --model $model \
                --batch_size 4 \
                --chunk_size 64 \
                --methods $method
        done

        python3 -m experiments.ablation.ablation_pipe \
            --ablation component \
            --type disrupt \
            --dataset "${dataset}"  \
            --model $model \
            --batch_size 64

        python3 -m experiments.ablation.ablation_pipe \
            --ablation component \
            --type recover \
            --dataset "${dataset}" \
            --model $model \
            --batch_size 64
    done
done