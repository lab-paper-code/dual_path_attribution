import os
# os.environ['CUDA_VISIBLE_DEVICES']="1"
import sys
import json
import hashlib
import argparse
import logging
import csv
import time
from typing import Dict, List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from nnsight import LanguageModel

# Import our modular methods
import experiments.attribution.input_attribution_modules as input_methods
import experiments.attribution.component_attribution_modules as component_methods
from experiments.attribution.utils import MODEL_CONFIGS, DATASET_CONFIGS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, model_id, dataset_name, tokenizer):
        self.tokenizer = tokenizer
        
        if model_id not in MODEL_CONFIGS:
            raise KeyError(f"Model '{model_id}' not in mapping.")
        self.insert_prefix_space = MODEL_CONFIGS[model_id]['insert_prefix_space']
        self.use_system_msg = MODEL_CONFIGS[model_id]['use_system_msg']

        if dataset_name not in DATASET_CONFIGS:
            raise KeyError(f"Dataset '{dataset_name}' not in mapping.")
        self._process_dataset = DATASET_CONFIGS[dataset_name]['process_fn']
    
    def get_target_ids(self, targets, tokenizer):
        encoded = []
        for target in targets:
            if self.insert_prefix_space:
                text = f" {target.strip()}"
            else:
                text = target.strip()
            encoded.append(tokenizer.encode(text, add_special_tokens=False)[0])
        return torch.tensor(encoded)

    def collate_fn(self, batch):
        """Shared tokenization and masking logic."""
        ids, prompts, contexts, answers = self._process_dataset(batch, self.tokenizer, self.use_system_msg)
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, return_offsets_mapping=True, add_special_tokens=False)
        
        # Context Masking Logic
        context_token_loc = []
        for i in range(len(ids)):
            # Simplified robust alignment
            start = prompts[i].find(contexts[i])
            end = start + len(contexts[i])
            offset = inputs['offset_mapping'][i]
            
            # Find token indices
            s_tok = next(k for k, (_, e) in enumerate(offset) if e > start)
            e_tok = next((k + 1 for k, (_, e) in enumerate(offset) if e >= end - 1))
            context_token_loc.append((s_tok, e_tok))

        context_token_loc = torch.tensor(context_token_loc)
        positions = torch.arange(inputs['input_ids'].size(1)).unsqueeze(0)
        context_mask = (positions >= context_token_loc[:, 0].unsqueeze(1)) & (positions < context_token_loc[:, 1].unsqueeze(1))

        return {
            'ids': ids, 'prompts': prompts, 'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'], 'context_mask': context_mask,
            'target_ids': self.get_target_ids(answers, self.tokenizer), 'target_str': answers
        }

# ==========================================
# Factory for Methods
# ==========================================
def get_input_attribution_methods(args, model, tokenizer):
    """Factory to instantiate requested methods."""
    active_methods = []
    
    # Mapping of CLI names to Classes
    if "random" in args.methods:
        active_methods.append(input_methods.RandomInputAttribution(model, tokenizer))
    
    if "last_layer_attn" in args.methods:
        active_methods.append(input_methods.LastLayerAttentionInputAttribution(model, tokenizer))

    if "mean_attn" in args.methods:
        active_methods.append(input_methods.MeanAttentionInputAttribution(model, tokenizer))

    if "attn_rollout" in args.methods:
        active_methods.append(input_methods.AttentionRolloutInputAttribution(model, tokenizer))

    if "gradient" in args.methods:
        active_methods.append(input_methods.GradientInputAttribution(model, tokenizer))
        
    if "input_x_gradient" in args.methods:
        active_methods.append(input_methods.InputXGradientInputAttribution(model, tokenizer))

    if "integrated_gradient" in args.methods:
        active_methods.append(input_methods.IntegratedGradientInputAttribution(model, tokenizer, args.ig_steps))
        
    if "depass" in args.methods:
        active_methods.append(input_methods.DePassInputAttribution(model, tokenizer, args.chunk_size))
    
    if "ifr" in args.methods:
        active_methods.append(input_methods.IFRInputAttribution(model, tokenizer, args.chunk_size))
    
    if "dpa" in args.methods:
        active_methods.append(input_methods.DPAInputAttribution(model, tokenizer, args.dpa_weights))

    if "ap" in args.methods:
        active_methods.append(input_methods.APInputAttribution(model, tokenizer, args.chunk_size))

    return active_methods

def get_component_attribution_methods(args, model, tokenizer):
    """Factory to instantiate requested methods."""
    active_methods = []
    
    # Mapping of CLI names to Classes
    if "random" in args.methods:
        active_methods.append(component_methods.RandomComponentAttribution(model, tokenizer))

    if "attn_act" in args.methods:
        active_methods.append(component_methods.AttnActComponentAttribution(model, tokenizer))

    if "mlp_act" in args.methods:
        active_methods.append(component_methods.MLPActComponentAttribution(model, tokenizer))

    if "norm" in args.methods:
        active_methods.append(component_methods.NormComponentAttribution(model, tokenizer))
    
    if "ifr" in args.methods:
        active_methods.append(component_methods.IFRComponentAttribution(model, tokenizer, args.chunk_size))
    
    if "gradient" in args.methods:
        active_methods.append(component_methods.GradienComponentAttribution(model, tokenizer))

    if "atp" in args.methods:
        active_methods.append(component_methods.AtPComponentAttribution(model, tokenizer))
    
    if 'depass' in args.methods:
        active_methods.append(component_methods.DepassComponentAttribution(model, tokenizer, args.chunk_size))

    if "ap" in args.methods:
        active_methods.append(component_methods.APComponentAttributionMethod(model, tokenizer, args.chunk_size))

    if "dpa" in args.methods:
        active_methods.append(component_methods.DPAComponentAttribution(model, tokenizer, args.dpa_weights))

    return active_methods


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_memory():
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)


def _get_peak_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peaks = []
    for device in range(torch.cuda.device_count()):
        peaks.append(torch.cuda.max_memory_allocated(device))
    if not peaks:
        return 0.0
    return sum(peaks) / (1024 ** 2)


def _get_peak_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    peaks = []
    for device in range(torch.cuda.device_count()):
        peaks.append(torch.cuda.max_memory_reserved(device))
    if not peaks:
        return 0.0
    return sum(peaks) / (1024 ** 2)

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution", default='input', choices=['input','component'])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", default="imdb_clean")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=512)

    input_attr_choices = ["random", "last_layer_attn", "mean_attn", "attn_rollout", "gradient", "input_x_gradient", "integrated_gradient", "depass", "ifr", "dpa", "ap"]
    component_attr_choices = ["random", "attn_act", "mlp_act", "norm", "gradient", "atp", "ifr", "dpa", "ap"]
    parser.add_argument("--methods", nargs="+", choices=list(set(input_attr_choices + component_attr_choices)), default=["random"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir", default="/mnt/dacslab/dpa/results")
    parser.add_argument("--force_data", action="store_true")

    parser.add_argument("--dpa_weights", nargs=5, type=float, default=[0.25, 0.25, 0.5, 0.5, 0.5])
    parser.add_argument("--ig_steps", type=int, default=20)

    args = parser.parse_args()

    args.dpa_weights = {'q':args.dpa_weights[0], 'k':args.dpa_weights[1], 'v':args.dpa_weights[2], 'gate':args.dpa_weights[3], 'up':args.dpa_weights[4]}
    

    args.save_dir = os.path.join(args.out_dir, f"attribution__{args.model.lower().replace('/','_').replace('-','_')}__{args.dataset}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading Model {args.model}")
    model = LanguageModel(args.model, attn_implementation= 'eager', device_map='auto', dispatch=True, dtype=torch.bfloat16)
    dh = DataHandler(args.model, args.dataset, tokenizer)

    data_path = os.path.join(args.data_dir, f"{args.dataset}.json")
    with open(data_path) as f: raw_data = json.load(f)

    if not args.model in raw_data[0].keys():
        probability_threshhold = 0.1
        
        dl = DataLoader(raw_data, collate_fn=dh.collate_fn, batch_size=args.batch_size, shuffle=False)

        valid_data = []
        for batch in tqdm(dl, desc='Checking knowledge on examples', total=len(dl), mininterval=5.0):
            with torch.no_grad(), model.trace(batch):
                next_token_logits = model.lm_head.output[:, -1]
                next_token_probs = torch.softmax(next_token_logits, dim=-1).save()
            
            context_not_too_long = batch['context_mask'].sum(-1) <= 400
            is_valid_target_token = torch.tensor([len(token) > 1 for token in tokenizer.convert_ids_to_tokens(batch['target_ids'])])
            next_token_is_target_token = torch.argmax(next_token_probs, dim=-1).cpu() == batch['target_ids']
            next_token_is_probable = next_token_probs[range(batch['target_ids'].size(0)), batch['target_ids']].cpu() > probability_threshhold

            valid_data.extend(
                (context_not_too_long & is_valid_target_token & next_token_is_probable & next_token_is_target_token).tolist()
            )
        
        for i in range(len(raw_data)):
            raw_data[i][args.model] = valid_data[i]

        with open(data_path, 'w') as f:
            json.dump(raw_data, f, indent=2)

    data = [sample for sample in raw_data if sample[args.model]]
    dl = DataLoader(data, collate_fn=dh.collate_fn, batch_size=args.batch_size, shuffle=False)

    data_path = os.path.join(args.save_dir, f"samples.json")
    if not os.path.exists(data_path) or args.force_data:

        sample_data = []
        for batch in dl:
            for i in range(len(batch['ids'])):

                tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i][batch['attention_mask'][i].bool()])
                context_mask = batch['context_mask'][i][batch['attention_mask'][i].bool()].int().tolist()
                context_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i][batch['context_mask'][i].bool()])

                assert len(tokens) == len(context_mask), f"'tokens' and 'context_maks' should be same length."
                assert len(context_tokens) == sum(context_mask), f"'context_tokens' should be of same length as 'context_maks' mask."

                sample_data.append({
                    'id': batch['ids'][i],
                    'prompts': batch['prompts'][i],
                    'tokens': tokens,
                    'context_mask': context_mask,
                    'context_tokens': context_tokens,
                    'target_str': batch['target_str'][i],
                    'target_ids': batch['target_ids'][i].item(),
                })

        data_hash = hashlib.sha256(json.dumps(raw_data, sort_keys=True).encode()).hexdigest()
        
        with open(data_path, 'w') as f: 
            json.dump({
                "metadata": {
                    "hash": data_hash,
                    "model": args.model,
                    "dataset": args.dataset,
                },
                "sample_data": sample_data
            }, f, indent=2)

    with open(data_path) as f: data_hash = json.load(f)['metadata']['hash']

    if args.attribution == 'input':
        active_methods = get_input_attribution_methods(args, model, tokenizer)
    if args.attribution == 'component':
        active_methods = get_component_attribution_methods(args, model, tokenizer)

    logger.info(f"Active Methods: {[m.name() for m in active_methods]}")

    for method in active_methods:
        _reset_peak_memory()
        _sync_cuda()
        method_start = time.perf_counter()
        scores = []
        for batch in tqdm(dl, desc=f"{method.name()}", total=len(dl), mininterval=5.0):
            batch_scores = method.run(batch)
            batch_scores = [{'id': id, 'scores': scores} for id, scores in zip(*batch_scores.values())]
            scores.extend(batch_scores)
        
        _sync_cuda()
        compute_elapsed = time.perf_counter() - method_start
        peak_allocated_mb = _get_peak_allocated_mb()
        peak_reserved_mb = _get_peak_reserved_mb()

        fname = os.path.join(args.save_dir, f"{args.attribution}_{method.name()}.pt")
        payload = {
            "metadata": {
                "parent_hash": data_hash,
                "model": args.model,
                "dataset": args.dataset,
                "method": method.name(),
                "batch_size": args.batch_size,
                "time_sec": round(compute_elapsed, 4),
                "peak_memory_allocated_mb": round(peak_allocated_mb, 2),
                "peak_memory_reserved_mb": round(peak_reserved_mb, 2),

            },
            "scores": scores
        }
        torch.save(payload, fname)
        logger.info(f"Saved {method.name} to {fname}")

if __name__ == "__main__":
    main()