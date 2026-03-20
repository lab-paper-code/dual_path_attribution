import torch

class DataCollator:

    def __init__(self, model, tokenizer, ):
        self.tokenizer = tokenizer

        self.num_hidden_layers = model.config.num_hidden_layers
        self.num_attention_heads = model.config.num_attention_heads
        self.intermediate_size = model.config.intermediate_size

    def __call__(self, batch):
        pass

class InputCollator(DataCollator):

    def __call__(self, batch):
        batch_samples, batch_scores = list(zip(*batch))
        prompts = [sample['prompts'] for sample in batch_samples]
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)

        batch_size, seq_len = inputs['input_ids'].shape
        context_mask = [sample['context_mask'] for sample in batch_samples]
        context_mask = torch.tensor([[0] * (seq_len - len(mask)) + mask for mask in context_mask])

        scores = torch.full((batch_size, seq_len), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)

        for i, score_seq in enumerate(batch_scores):
            scores[i][context_mask[i].bool()] = score_seq['scores'].to(torch.bfloat16)

        inputs['ids'] = [sample['id'] for sample in batch_samples]
        inputs['scores'] = scores
        inputs['context_mask'] = context_mask
        inputs['num_nodes'] = context_mask.sum(-1)
        inputs['target_ids'] = [sample['target_ids'] for sample in batch_samples]

        return inputs

class ComponentCollator(DataCollator):
        
        def __call__(self, batch):
            batch_samples, batch_scores = list(zip(*batch))
            prompts = [sample['prompts'] for sample in batch_samples]
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)

            batch_size, seq_len = inputs['input_ids'].shape
            context_mask = [sample['context_mask'] for sample in batch_samples]
            context_mask = torch.tensor([[0] * (seq_len - len(mask)) + mask for mask in context_mask])
            attn_context_mask = (context_mask[:, None, :] * context_mask[:, :, None])

            attn_scores = torch.full((batch_size, seq_len, seq_len, self.num_hidden_layers, self.num_attention_heads), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
            mlp_scores = torch.full((batch_size, seq_len, self.num_hidden_layers, self.intermediate_size), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16)
            for i, score_seq in enumerate(batch_scores):
                attn_seq, mlp_seq = score_seq['scores']
                
                if attn_seq != None:
                    attn_scores[i, attn_context_mask[i].bool()] = attn_seq.flatten(0, 1).to(torch.bfloat16)
                else:
                    attn_scores = None
                if mlp_seq != None:
                    mlp_scores[i, context_mask[i].bool()] = mlp_seq.to(torch.bfloat16)
                else:
                    mlp_scores = None

            contex_len = context_mask.sum(-1)
            if attn_seq != None:
                num_attn_nodes = contex_len * contex_len * self.num_hidden_layers * self.num_attention_heads
            else:
                num_attn_nodes = 0

            if mlp_seq != None:
                num_mlp_nodes = contex_len * self.num_hidden_layers * self.intermediate_size
            else:
                num_mlp_nodes = 0

            inputs['ids'] = [sample['id'] for sample in batch_samples]
            inputs['scores'] = (attn_scores, mlp_scores)
            inputs['context_mask'] = context_mask
            inputs['num_nodes'] = num_attn_nodes + num_mlp_nodes
            inputs['target_ids'] = [sample['target_ids'] for sample in batch_samples]
        
            return inputs
        

DATASET_CONFIG = {
    "known_1000_input": {
        "percentages": [i/10 for i in range(1, 11)]
    },
    "imdb_input": {
        "percentages": torch.sigmoid(torch.arange(-6, 7, 1.2)).tolist()
    },
    "squad_v2.0_input": {
        "percentages": torch.sigmoid(torch.arange(-6, 7, 1.2)).tolist()
    },
    "known_1000_component": {
        "percentages": torch.exp(torch.arange(-17, 1, 1.8)).tolist()
    },
    "ioi_component": {
        "percentages": torch.exp(torch.arange(-17, 1, 1.8)).tolist()
    },
}