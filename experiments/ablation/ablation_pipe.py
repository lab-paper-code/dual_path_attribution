import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import logging
import json
from tqdm import tqdm

from torch.utils.data import DataLoader
from experiments.ablation.utils import InputCollator, ComponentCollator, DATASET_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AblationProcessor:

    def __init__(self, model, type, percentages):
        self.model = model
        self.type = type
        self.percentages = percentages

        self.device = model.device

    def run(self, batch):
        pass

class InputAblationProcessor(AblationProcessor):

    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape

        with torch.no_grad(), self.model.trace(batch):
            logits = self.model.lm_head.output[:, -1]
            base_probs = torch.softmax(logits, dim=-1)[range(batch_size), batch['target_ids']].cpu().save()

        sorted_score_idx = torch.argsort(batch['scores'], dim=-1, descending=True)
        range_tensor = torch.arange(seq_len).expand(batch_size, seq_len)

        scores = {}
        for percentage in self.percentages:
            num_nodes_in_top_x = (batch['num_nodes'] * percentage).ceil().long()
            mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
            
            if self.type == 'disrupt':
                mask_indices = range_tensor < num_nodes_in_top_x.unsqueeze(1)
                mask.scatter_(dim=1, index=sorted_score_idx, src=mask_indices)
            if self.type == 'recover':
                mask_indices = range_tensor >= num_nodes_in_top_x.unsqueeze(1)
                mask.scatter_(dim=1, index=sorted_score_idx, src=mask_indices)
                mask = mask & batch['context_mask'].bool()
            
            with torch.no_grad(), self.model.trace(batch):
                self.model.model.embed_tokens.output[mask.to(self.device)] = 0

                logits = self.model.lm_head.output[:, -1]
                ablated_probs = torch.softmax(logits, dim=-1)[range(batch_size), batch['target_ids']].cpu().save()

            scores[f"noise_{percentage}"] = ((base_probs - ablated_probs) / base_probs * 100).tolist()

        return {
            "id": batch['ids'],
            "scores": scores
        }

class ComponentAblationProcessor(AblationProcessor):

    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape
        with torch.no_grad(), self.model.trace(batch):
            logits = self.model.lm_head.output[:, -1]
            base_probs = torch.softmax(logits, dim=-1)[range(batch_size), batch['target_ids']].cpu().save()

        attn_scores, mlp_scores = batch['scores']
        non_null_scores = []
        if attn_scores is not None:
            non_null_scores.append(attn_scores)
            attn_sh = attn_scores.shape
            num_flat_attn_el = attn_scores.view(batch_size, -1).shape[1]
        if mlp_scores is not None:
            non_null_scores.append(mlp_scores)
            mlp_sh = mlp_scores.shape

        sorted_score_idx = torch.argsort(
            torch.concat([scores.view(batch_size, -1) for scores in non_null_scores], dim=-1)
        , dim=-1, descending=True)
        total_features = sorted_score_idx.shape[1]
        range_tensor = torch.arange(total_features).expand(batch_size, -1)

        scores = {}
        for percentage in self.percentages:
            num_nodes_in_top_x = (batch['num_nodes'] * percentage).ceil().long()
            global_mask = torch.zeros((batch_size, total_features), dtype=torch.bool)

            if self.type == 'disrupt':
                mask_indices = range_tensor < num_nodes_in_top_x.unsqueeze(1)
                global_mask.scatter_(dim=1, index=sorted_score_idx, src=mask_indices)

            if self.type == 'recover':
                mask_indices = range_tensor >= num_nodes_in_top_x.unsqueeze(1)
                global_mask.scatter_(dim=1, index=sorted_score_idx, src=mask_indices)
            
            attn_mask = None
            mlp_mask = None
            split_idx = 0

            if attn_scores is not None:
                flat_attn_mask = global_mask[:, :num_flat_attn_el]
                attn_mask = flat_attn_mask.view(attn_sh)
                if self.type == 'recover':  
                    attn_mask = attn_mask & (batch['context_mask'][:, None, :] * batch['context_mask'][:, :, None])[:,:,:,None,None].bool()
                attn_mask = attn_mask.permute(3, 0, 4, 1, 2)
                split_idx = num_flat_attn_el
            
            if mlp_scores is not None:
                flat_mlp_mask = global_mask[:, split_idx:]
                mlp_mask = flat_mlp_mask.view(mlp_sh)
                if self.type == 'recover':  
                    mlp_mask = mlp_mask & batch['context_mask'][:, :, None, None].bool()
                mlp_mask = mlp_mask.permute(2, 0, 1, 3)

            with torch.no_grad(), self.model.trace(batch):
                for layer_id, layer in enumerate(self.model.model.layers):
                    if attn_mask is not None:
                        layer.self_attn.source.attention_interface_0.source.nn_functional_softmax_0.output[attn_mask[layer_id]] = 0
                    if mlp_mask is not None:
                        layer.mlp.act_fn.output[mlp_mask[layer_id]] = 0

                logits = self.model.lm_head.output[:, -1]
                ablated_probs = torch.softmax(logits, dim=-1)[range(batch_size), batch['target_ids']].cpu().save()

            scores[f"noise_{percentage}"] = ((base_probs - ablated_probs) / base_probs * 100).tolist()

        return {
            "id": batch['ids'],
            "scores": scores
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default='input', choices=['input','component'])
    parser.add_argument("--type", default='disrupt', choices=['disrupt','recover'])
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", default="imdb_clean")
    parser.add_argument("--batch_size", type=int, default=64)

    input_attr_choices = ["random", "last_layer_attn", "mean_attn", "attn_rollout", "gradient", "input_x_gradient", "integrated_gradient", "depass", "ifr", "dpa"]
    component_attr_choices = ["random", "attn_act", "mlp_act", "norm", "gradient", "atp", "ifr", "dpa_025_025_05_05_05", "dpa_star"] # TODO fix dpa
    parser.add_argument("--methods", nargs="+", default=[]) # choices=list(set(input_attr_choices + component_attr_choices))
    parser.add_argument("--attr_dir", default="/mnt/dacslab/dpa/results")
    parser.add_argument("--out_dir", default="/mnt/dacslab/dpa/results")

    args = parser.parse_args()

    args.load_dir = os.path.join(args.out_dir, f"attribution__{args.model.lower().replace('/','_').replace('-','_')}__{args.dataset}")
    
    if not args.methods:
        methods = []
        for fname in os.listdir(args.load_dir):
            if fname.startswith(args.ablation):
                methods.append(fname.removeprefix(f"{args.ablation}_").removesuffix('.pt'))
        args.methods = methods

    args.save_dir = os.path.join(args.out_dir, f"ablation__{args.model.lower().replace('/','_').replace('-','_')}__{args.dataset}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading Model {args.model}")
    model = LanguageModel(args.model, attn_implementation= 'eager', device_map='auto', dispatch=True, dtype=torch.bfloat16)

    with open(os.path.join(args.load_dir, 'samples.json'), 'r') as f:
        sample_data = json.load(f)
    
    samples = sample_data['sample_data']
    data_hash = sample_data['metadata']['hash']

    percentages = DATASET_CONFIG[f"{args.dataset}_{args.ablation}"]['percentages']

    if args.ablation == 'input':
        collate_fn = InputCollator(model, tokenizer)
        processor = InputAblationProcessor(model, args.type, percentages)
    if args.ablation == 'component':
        collate_fn = ComponentCollator(model, tokenizer)
        processor = ComponentAblationProcessor(model, args.type, percentages)

    for method in args.methods:

        score_data = torch.load(os.path.join(args.load_dir, f"{args.ablation}_{method}.pt"))
        scores = score_data['scores']
        score_hash = score_data['metadata']['parent_hash']

        if not data_hash == score_hash:
            logger.warning(f"The sample and the attribution data might not fit. Did you change the sample.json after you ran the {method} attribution?")
        
        data = list(zip(samples, scores))
        dl = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    
        scores = []
        for batch in tqdm(dl, desc=f"{method}", total=len(dl), mininterval=5.0):
            batch_result = processor.run(batch)
            batch_scores = []
            for i in range(len(batch_result['id'])):
                batch_scores.append({
                    'id': batch_result['id'][i],
                    'scores': {
                        k: v[i] for k, v in batch_result['scores'].items()
                    }
                })
            scores.extend(batch_scores)
        
        fname = f"{args.ablation}_{args.type}_{method}.json"
        with open(os.path.join(args.save_dir, fname), 'w') as f:
            json.dump(scores, f, indent=2)

if __name__ == '__main__':
    main()