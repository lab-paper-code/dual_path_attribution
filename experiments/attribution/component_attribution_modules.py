import torch
import gc
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import random
import copy
from tqdm import tqdm

from tracer.tracer import ComponentTracer
from tracer.backend import BACKEND_MAPPING



################################################################
#                       Abstract Classes                       #
################################################################

class ComponentAttributionMethod(ABC):
    """Abstract base class for all attribution methods."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, 'device') else 'cuda'
        self.dtype = self.model.dtype

    @abstractmethod
    def run(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes the attribution method on a batch.
        Returns a list of dictionaries containing 'id' and the calculated scores.
        """
        raise NotImplementedError
    
    @abstractmethod
    def name(self):
        """
        Returns name of method.
        """
        pass

    def _extract_from_mask(self, tensors, mask):
        """Helper to extract context-only tokens from flat tensors based on masks."""
        # This handles the logic of unflattening the nnsight output
        attn_tensor, mlp_tensor = tensors
        if attn_tensor == None and mlp_tensor == None:
            return [None, None]
        
        length = mask.sum(dim=1)

        # extract attn contributions
        if attn_tensor != None:
            num_layer, num_heads = attn_tensor.shape[-2:]
            attn_mask = (mask[:, None, :] * mask[:, :, None]).bool()
            flat_attn_tensor= attn_tensor[attn_mask.to(attn_tensor.device)]
            attn_lists = [
                t.view(l, l, num_layer, num_heads).cpu() for t, l in zip(
                    torch.split(flat_attn_tensor, (length ** 2).tolist()), length.tolist()
                )
            ]

        # extract mlp contributions
        if mlp_tensor != None:
            flat_mlp_tensor= mlp_tensor[mask.bool().to(mlp_tensor.device)]
            mlp_lists = [t.cpu() for t in torch.split(flat_mlp_tensor, length.tolist())]

        if attn_tensor == None:
            attn_lists = [None for _ in range(len(mlp_lists))]
        if mlp_tensor == None:
            mlp_lists = [None for _ in range(len(attn_lists))]

        return list(zip(attn_lists, mlp_lists))
    

class DecompositionAttributionMethod(ComponentAttributionMethod):

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.num_attention_heads = self.model.config.num_attention_heads
        self.num_key_value_heads= self.model.config.num_key_value_heads
        if not hasattr(model.config, 'head_dim') or model.config.head_dim is None:
            model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = model.config.head_dim

        self.num_head_groups = self.num_attention_heads // self.num_key_value_heads

    def decompose_attention_to_head(
            self,
            attn_weight: torch.Tensor, 
            v_proj: torch.Tensor, 
            o_proj_WT: torch.Tensor,
            batch_token_index: Tuple[List[int], List[int]] = None
    ) -> torch.Tensor:
        """
        Decomposes attention output into contributions from each head.
        
        Args:
            attn_weight: (bs, num_heads, q_pos, k_pos)
            v_proj: (bs, k_pos, num_kv_heads * head_dim)
            o_proj_WT: (num_heads * head_dim, model_dim) - Assumed to be (In, Out)
        """
        batch_size = v_proj.size(0)

        # Reshape v_poj to k_v_head view (bs, k_pos, num_k_v_heads, head_dim)
        v_proj = v_proj.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
        # Reshape o_proj to head view (num_heads, head_dim, model_dim)
        o_proj_WT = o_proj_WT.view(self.num_attention_heads, self.head_dim, -1)
        
        if self.num_head_groups > 1:
            # Repeat v_poj for number of head groups (bs, k_pos, num_heads, head_dim)
            v_proj = v_proj.repeat_interleave(self.num_head_groups, -2)
            
        if batch_token_index != None:
            batch_idx, token_idx = batch_token_index
            # select target (bs*, k_pos, num_heads, head_dim)
            attn_weight = attn_weight[batch_idx, : ,token_idx]
            v_proj = v_proj[batch_idx]
            
            z = torch.einsum('bkhd,bhk -> bhkd', v_proj, attn_weight)
            decomposed_attn = torch.einsum('bhkd,hdm -> bkhm', z, o_proj_WT)

            return decomposed_attn
            
        z = torch.einsum('bkhd,bhqk -> bhqkd', v_proj, attn_weight)
        decomposed_attn = torch.einsum('bhqkd,hdm -> bqkhm',z, o_proj_WT)
        
        return decomposed_attn

    def decompose_glu_to_neuron(
        self,
        down_proj_WT: torch.Tensor, 
        act_prod: torch.Tensor = None,
        batch_token_index: Tuple[List[int], List[int]] = None
        ):
        """
        Decomposes GLU/MLP output into contributions from each intermediate neuron.
        
        Args:
            down_proj_WT: (intermediate_dim, model_dim)
            act_prod: (bs, seq_len, intermediate_dim) - This is (Gate * Up) output
        """
        if batch_token_index != None:
            batch_idx, token_idx = batch_token_index
            act_prod = act_prod[batch_idx, token_idx]

            decomposed_glu = torch.einsum('bh,hm->bhm', act_prod, down_proj_WT)

            return decomposed_glu
        
        decomposed_glu = torch.einsum('bth,hm->bthm', act_prod, down_proj_WT)
        
        return decomposed_glu


################################################################
#                       Random Baseline                        #
################################################################


class RandomComponentAttribution(ComponentAttributionMethod):

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.max_len = 0
        self.master_flat = None
        
        self.num_hidden_layers = model.config.num_hidden_layers
        self.num_attention_heads = model.config.num_attention_heads
        self.intermediate_size = model.config.intermediate_size

    def name(self):
        return "random"

    def run(self, batch):
        context_lengths = batch['context_mask'].sum(dim=1).tolist()
        current_max_len = max(context_lengths)
        
        if current_max_len > self.max_len:

            self.max_len = current_max_len
            
            max_attn_size = self.num_hidden_layers * self.num_attention_heads * (self.max_len ** 2)
            max_mlp_size = self.num_hidden_layers * self.max_len * self.intermediate_size
            required_size = max(max_attn_size, max_mlp_size)
            
            self.master_flat = torch.randn(required_size, dtype=torch.float32)

        master_len = len(self.master_flat)
        scores = []
        for length in context_lengths:

            # Attn. Scores
            num_attn_el = self.num_hidden_layers * self.num_attention_heads * length * length
            attn_start_idx = random.randint(0, master_len - num_attn_el)
            attn_scores = self.master_flat[attn_start_idx : attn_start_idx + num_attn_el].view(
                length, length, self.num_hidden_layers, self.num_attention_heads
            )

            # MLP Scores
            num_mlp_el = self.num_hidden_layers * length * self.intermediate_size
            mlp_start_idx = random.randint(0, master_len - num_mlp_el)
            mlp_scores = self.master_flat[mlp_start_idx : mlp_start_idx + num_mlp_el].view(
                length, self.num_hidden_layers, self.intermediate_size
            )
            
            scores.append((attn_scores, mlp_scores))

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    


################################################################
#                     Activation Baselines                     #
################################################################

class AttnActComponentAttribution(ComponentAttributionMethod):

    def name(self):
        return 'attn_act'
    
    def run(self, batch):
        batch_attn = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:
                batch_attn.append(
                    layer.self_attn.output[1].save()
                )
        batch_attn = torch.stack([t.to(self.device) for t in batch_attn]).permute(1, 3, 4, 0, 2).cpu() # (B, Q, K, L, H)
        scores = self._extract_from_mask((batch_attn, None), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }

class MLPActComponentAttribution(ComponentAttributionMethod):

    def name(self):
        return 'mlp_act'
    
    def run(self, batch):
        batch_mlp = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:
                batch_mlp.append(
                    layer.mlp.act_fn.output.save()
                )
        batch_mlp = torch.stack([t.to(self.device) for t in batch_mlp]).permute(1, 2, 0, 3).cpu() # (B, Q, L, D)
        scores = self._extract_from_mask((None, batch_mlp), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
################################################################
#                    Gradient Baseline                         #
################################################################


class GradienComponentAttribution(ComponentAttributionMethod):

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.num_layers = model.config.num_hidden_layers

    def name(self):
        return 'gradient'
    
    def run(self, batch):
        curr_bs = batch['input_ids'].size(0)

        with self.model.trace(batch):

            attn_weights = []
            act_prods = []
            for layer in self.model.model.layers:
                attn_weights.append(layer.self_attn.output[1])
                act_prods.append(layer.mlp.down_proj.input)

            logits = self.model.lm_head.output[:, -1]
            target_logits = logits[range(curr_bs), batch['target_ids']]

            grads = torch.autograd.grad(
                outputs=target_logits,
                inputs=attn_weights + act_prods,
                grad_outputs=torch.ones_like(target_logits),
            )

            attn_grads = grads[:self.num_layers].save()
            mlp_grads = grads[self.num_layers:].save()

        attn_grads = torch.stack([t.to(self.device) for t in attn_grads]).permute(1, 3, 4, 0, 2).cpu() # (B, Q, K, L, H)
        mlp_grads = torch.stack([t.to(self.device) for t in mlp_grads]).permute(1, 2, 0, 3).cpu() # (B, Q, I)

        scores = self._extract_from_mask((attn_grads, mlp_grads), batch['context_mask'])
        
        return {
            'ids': batch['ids'],
            'scores': scores
        }

class AtPComponentAttribution(ComponentAttributionMethod):

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.num_layers = model.config.num_hidden_layers

    def name(self):
        return 'atp'
    
    def run(self, batch):
        curr_bs = batch['input_ids'].size(0)

        attn_weights = []
        act_prods = []
        with self.model.trace(batch):
            for layer in self.model.model.layers:
                attn_weights.append(layer.self_attn.output[1]).save()
                act_prods.append(layer.mlp.down_proj.input).save()

            logits = self.model.lm_head.output[:, -1]
            target_logits = logits[range(curr_bs), batch['target_ids']]

            grads = torch.autograd.grad(
                outputs=target_logits,
                inputs=attn_weights + act_prods,
                grad_outputs=torch.ones_like(target_logits),
            )

            attn_grads = grads[:self.num_layers].save()
            mlp_grads = grads[self.num_layers:].save()

        attn_act = torch.stack([t.to(self.device) for t in attn_weights]).permute(1, 3, 4, 0, 2).cpu() # (B, Q, K, L, H)
        mlp_act = torch.stack([t.to(self.device) for t in act_prods]).permute(1, 2, 0, 3).cpu() # (B, Q, I)

        attn_grads = torch.stack([t.to(self.device) for t in attn_grads]).permute(1, 3, 4, 0, 2).cpu() # (B, Q, K, L, H)
        mlp_grads = torch.stack([t.to(self.device) for t in mlp_grads]).permute(1, 2, 0, 3).cpu() # (B, Q, I)

        scores = self._extract_from_mask((attn_act * attn_grads, mlp_act * mlp_grads), batch['context_mask'])
        
        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
################################################################
#                    Decompostion Baseline                     #
################################################################

class NormComponentAttribution(DecompositionAttributionMethod):

    def name(self):
        return 'norm'
    
    def run(self, batch):
        batch_attn_norm = []
        batch_mlp_norm = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:

                # Attn Norms
                v_proj = layer.self_attn.v_proj.output
                attn_weight = layer.self_attn.output[1]
                o_proj_WT = layer.self_attn.o_proj.weight.data.T
                d_attn = self.decompose_attention_to_head(
                    attn_weight,
                    v_proj,
                    o_proj_WT,
                ) # (B, Q, K, H, D)
                batch_attn_norm.append(
                    d_attn.norm(dim=-1).save() # (B, Q, K, H)
                )
                del d_attn

                act_prod = layer.mlp.down_proj.input
                down_proj_WT = layer.mlp.down_proj.weight.data.T
                d_mlp = self.decompose_glu_to_neuron(
                    down_proj_WT,
                    act_prod
                ) # (B, Q, I, D)
                batch_mlp_norm.append(
                    d_mlp.norm(dim=-1).save() # (B, Q, I)
                )
                del d_mlp

        batch_attn_norm = torch.stack([t.to(self.device) for t in batch_attn_norm]).permute(1, 2, 3, 0, 4).cpu() # (B, Q, K, L, H)
        batch_mlp_norm = torch.stack([t.to(self.device) for t in batch_mlp_norm]).permute(1, 2, 0, 3).cpu() # (B, Q, L, I)

        scores = self._extract_from_mask((batch_attn_norm, batch_mlp_norm), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }

class IFRComponentAttribution(DecompositionAttributionMethod):

    def __init__(self, model, tokenizer, chunk_size):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        self.intermediate_size = self.model.config.intermediate_size

    def name(self):
        return 'ifr'
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape
        context_mask_idx = batch['attention_mask'].nonzero()
        batch_idx, token_idx = context_mask_idx.T
        num_elements = len(batch_idx)

        attn_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_hidden_layers, self.num_attention_heads, dtype=self.dtype)
        mlp_scores = torch.zeros(batch_size, seq_len, self.num_hidden_layers, self.intermediate_size, dtype=self.dtype)

        with torch.no_grad(), self.model.trace(batch):
            for layer_id, layer in enumerate(self.model.model.layers):

                # Cache activation values
                v_proj = layer.self_attn.v_proj.output
                attn_out, attn_weight = layer.self_attn.output
                o_proj_WT = layer.self_attn.o_proj.weight.data.T

                act_prod = layer.mlp.down_proj.input
                mlp_out = layer.mlp.down_proj.output
                down_proj_WT = layer.mlp.down_proj.weight.data.T

                for start_idx in range(0, num_elements, self.chunk_size):
                    chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                    chunk_batch_idx, token_idx = context_mask_idx[chunk_slice].T

                    d_attn = self.decompose_attention_to_head(
                        attn_weight,
                        v_proj,
                        o_proj_WT,
                        batch_token_index=(chunk_batch_idx, token_idx)
                    ) # (BQ, K, H, D)

                    d_attn -= attn_out[chunk_batch_idx, token_idx][:, None, None, :]
                    layer_attn_scores = (-torch.norm(d_attn, p=1, dim=-1) + torch.norm(attn_out[chunk_batch_idx, token_idx], p=1, dim=-1)[:, None, None]).clip(min=0)
                    del d_attn
                    layer_attn_scores /= (layer_attn_scores.sum((-2, -1), keepdim=True) + 1e-7)
                    attn_scores[chunk_batch_idx, token_idx, :, layer_id] = layer_attn_scores.cpu()

                    d_mlp = self.decompose_glu_to_neuron(
                        down_proj_WT,
                        act_prod,
                        batch_token_index=(chunk_batch_idx, token_idx)
                    ) # (BQ, I, D)
                
                    d_mlp -= mlp_out[chunk_batch_idx, token_idx][:, None, :]
                    layer_mlp_scores = (-torch.norm(d_mlp, p=1, dim=-1) + torch.norm(mlp_out[chunk_batch_idx, token_idx], p=1, dim=-1)[:, None]).clip(min=0)
                    del d_mlp
                    layer_mlp_scores /= (layer_mlp_scores.sum(-1, keepdim=True) + 1e-7)
                    mlp_scores[chunk_batch_idx, token_idx, layer_id] = layer_mlp_scores.cpu()

            logits = self.model.lm_head.output[range(batch_size), -1, batch['target_ids']].save()

        grad = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
        grad[:, -1] = logits

        for layer_id in range(self.num_hidden_layers - 1, -1, -1):
            mlp_scores[:, :, layer_id] = (mlp_scores[:, :, layer_id].to(grad.device) * grad[:, :, None]).cpu()
            attn_scores[:, :, :, layer_id] = (attn_scores[:, :, :, layer_id].to(grad.device) * grad[:, :, None, None]).cpu()

            grad = attn_scores[:, :, :, layer_id].sum((-3, -1))

        scores = self._extract_from_mask((attn_scores, mlp_scores), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
class DepassComponentAttribution(ComponentAttributionMethod):

    def __init__(self, model, tokenizer, chunk_size):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size

        self.num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        if not hasattr(model.config, 'head_dim') or model.config.head_dim is None:
            model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim =  model.config.head_dim
        self.intermediate_size = self.model.config.intermediate_size

    def name(self):
        return 'depass'
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape

        causal_mask = self.get_causal_mask(batch['attention_mask'])
        attn_mask = causal_mask.repeat(1, self.num_attention_heads, 1, 1)
        attn_abl_idx = attn_mask.nonzero().cpu()
        num_attn_elements = attn_abl_idx.size(0)

        mlp_mask = batch['attention_mask'][:,:, None].repeat(1, 1, self.intermediate_size)
        mlp_abl_idx = mlp_mask.nonzero().cpu()
        num_mlp_elements = mlp_abl_idx.size(0)

        attn_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_hidden_layers, self.num_attention_heads, dtype=self.dtype)
        mlp_scores = torch.zeros(batch_size, seq_len, self.num_hidden_layers, self.intermediate_size, dtype=self.dtype)

        with torch.no_grad():

            cache = {}
            with self.model.trace(batch) as tracer:
                hidden_states = self.model.model.embed_tokens.output.save()
                args, kwargs = copy.deepcopy(self.model.model.layers[0].inputs)
                attention_mask = kwargs['attention_mask'].save() 
                position_embeddings = kwargs['position_embeddings'].save() 

                for layer_id, layer in enumerate(self.model.model.layers):

                    cache[f"attn.{layer_id}.pre_norm"] = self.get_rmsnorm_scaling(layer.input_layernorm.input).save()
                    cache[f"attn.{layer_id}.weights"] = layer.self_attn.output[1].save()

                    cache[f"mlp.{layer_id}.pre_norm"] = self.get_rmsnorm_scaling(layer.post_attention_layernorm.input).save()
                    cache[f"mlp.{layer_id}.z"] = layer.mlp.down_proj.input.save()
                
                norm_scalar = self.get_rmsnorm_scaling(self.model.model.norm.input).save()
            
            for layer_id, layer in enumerate(self.model.model.layers):

                # calculate attn ablations
                for start_idx in range(0, num_attn_elements, self.chunk_size):
                    chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                    chunk_batch_idx, head_idx, q_idx, k_idx = attn_abl_idx[chunk_slice].T
                    chunk_len = len(chunk_batch_idx)

                    ablation_mask = torch.zeros(chunk_len, self.num_attention_heads, seq_len, seq_len).bool()
                    ablation_mask[range(chunk_len), head_idx, q_idx, k_idx] = True
                
                    residual_batch_idx = torch.unique(chunk_batch_idx)
                    num_batch_idx = len(residual_batch_idx)
                    residual_mask = causal_mask.clone().cpu().repeat(1, self.num_attention_heads, 1, 1)
                    residual_mask[chunk_batch_idx, head_idx, q_idx, k_idx] = 0
                    residual_mask = residual_mask[residual_batch_idx]
                    batch_idx_needs_residual = residual_mask.any(dim=(1,2,3))
                    residual_batch_idx = residual_batch_idx[batch_idx_needs_residual]
                    residual_mask = residual_mask[batch_idx_needs_residual]

                    full_mask = ~torch.concat([ablation_mask, residual_mask]).bool() # concat and invert mask
                    full_chunk_batch_idx = torch.concat([chunk_batch_idx, residual_batch_idx])
                    inverse_chunk_batch_idx = torch.unique(full_chunk_batch_idx, return_inverse=True)[1]

                    chunk_mask = attention_mask[full_chunk_batch_idx]
                    chunk_states = hidden_states[full_chunk_batch_idx]

                    with layer.trace(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings):
                        layer.self_attn.source.attention_interface_0.source.nn_functional_dropout_0.output[full_mask] = 0
                        chunk_states = layer.self_attn.output[0].save()

                    chunk_states = self.process_decomposed_mlp(layer, layer_id, chunk_states, cache, full_chunk_batch_idx, inverse_chunk_batch_idx, num_batch_idx)
                    
                    for c_layer_id, c_layer in enumerate(self.model.model.layers[layer_id + 1:]):
                        chunk_states = self.process_decomposed_attn(c_layer, c_layer_id, chunk_states, cache, full_chunk_batch_idx)
                        chunk_states = self.process_decomposed_mlp(c_layer, c_layer_id, chunk_states, cache, full_chunk_batch_idx, inverse_chunk_batch_idx, num_batch_idx)

                    chunk_states = chunk_states * self.model.model.norm.weight.data * norm_scalar[full_chunk_batch_idx]
                    logits = self.model.lm_head(chunk_states)[range(chunk_len), -1, batch['target_ids'][chunk_batch_idx].tolist()]

                    attn_scores[chunk_batch_idx, q_idx, k_idx, layer_id, head_idx] = logits.cpu()

                
                # calculate mlp ablations
                for start_idx in tqdm(range(0, num_mlp_elements, self.chunk_size), desc=f"{layer_id}/{self.num_hidden_layers}"):
                    chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                    chunk_batch_idx, s_idx, n_idx = mlp_abl_idx[chunk_slice].T
                    chunk_len = len(chunk_batch_idx)

                    ablation_mask = torch.zeros(chunk_len, seq_len, self.intermediate_size).bool()
                    ablation_mask[range(chunk_len), s_idx, n_idx] = True
                
                    residual_batch_idx = torch.unique(chunk_batch_idx)
                    num_batch_idx = len(residual_batch_idx)
                    residual_mask = batch['attention_mask'].clone()[:, :, None].repeat(1, 1, self.intermediate_size)
                    residual_mask[chunk_batch_idx, s_idx, n_idx] = 0
                    residual_mask = residual_mask[residual_batch_idx]
                    batch_idx_needs_residual = residual_mask.any(dim=(1,2))
                    residual_batch_idx = residual_batch_idx[batch_idx_needs_residual]
                    residual_mask = residual_mask[batch_idx_needs_residual]

                    full_mask = ~torch.concat([ablation_mask, residual_mask]).bool() # concat and invert mask
                    full_chunk_batch_idx = torch.concat([chunk_batch_idx, residual_batch_idx])
                    inverse_chunk_batch_idx = torch.unique(full_chunk_batch_idx, return_inverse=True)[1]

                    chunk_mask = attention_mask[full_chunk_batch_idx]
                    chunk_states = hidden_states[full_chunk_batch_idx]

                    with layer.trace(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings):
                        layer.mlp.act_fn.output[full_mask] = 0
                        chunk_states = layer.mlp.output.save()
                    
                    for c_layer_id, c_layer in enumerate(self.model.model.layers[layer_id + 1:]):
                        chunk_states = self.process_decomposed_attn(c_layer, c_layer_id, chunk_states, cache, full_chunk_batch_idx)
                        chunk_states = self.process_decomposed_mlp(c_layer, c_layer_id, chunk_states, cache, full_chunk_batch_idx, inverse_chunk_batch_idx, num_batch_idx)

                    chunk_states = chunk_states * self.model.model.norm.weight.data * norm_scalar[full_chunk_batch_idx]
                    logits = self.model.lm_head(chunk_states)[range(chunk_len), -1, batch['target_ids'][chunk_batch_idx].tolist()]

                    mlp_scores[chunk_batch_idx, s_idx, layer_id, n_idx] = logits.cpu()
                
                hidden_states = layer(hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
                
        scores = self._extract_from_mask((attn_scores, mlp_scores), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }

    def get_causal_mask(self, attention_mask):
        batch_size, seq_len = attention_mask.shape

        mask = torch.ones((seq_len, seq_len), device=self.device, dtype=self.dtype)
        mask = torch.tril(mask).bool()
        padding_mask = torch.ones((batch_size, 1, 1, seq_len), device=self.device, dtype=self.dtype)
        padding_mask = padding_mask.masked_fill(attention_mask[:, None, None, :].to(self.device) == 0, 0).bool()
        causal_mask = mask & padding_mask
        return causal_mask.int()

    @staticmethod
    def get_rmsnorm_scaling(
        residual_state: torch.Tensor
    ) -> torch.Tensor:
        # transfer to float32 to avoid rounding errors during exponentiation
        input_dtype = residual_state.dtype
        residual_state = residual_state.to(torch.float32)

        variance = residual_state.pow(2).mean(-1, keepdim=True)
        rmsnorm_salar = torch.rsqrt(variance + 1e-7) # 1 / sqrt(x)
        return rmsnorm_salar.to(input_dtype)

    def process_decomposed_attn(self, layer, layer_id, hidden_states, cache, full_chunk_batch_idx):
        batch_size, seq_len = hidden_states.shape[:2]
        
        residual = hidden_states
        hidden_states = hidden_states * layer.input_layernorm.weight.data * cache[f"attn.{layer_id}.pre_norm"][full_chunk_batch_idx]
        v_proj = layer.self_attn.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        if self.num_key_value_groups > 1:
            v_proj = v_proj.repeat_interleave(self.num_key_value_groups, dim=1)
        hidden_states = torch.matmul(cache[f'attn.{layer_id}.weights'][full_chunk_batch_idx], v_proj).transpose(1, 2).flatten(2).contiguous()
        hidden_states = layer.self_attn.o_proj(hidden_states)\
        
        return residual + hidden_states

    def process_decomposed_mlp(self, layer, layer_id, hidden_states, cache, full_chunk_batch_idx, inverse_chunk_batch_idx, num_batch_idx):
        batch_size, seq_len = hidden_states.shape[:2]

        residual = hidden_states
        hidden_states = hidden_states * layer.post_attention_layernorm.weight.data * cache[f"mlp.{layer_id}.pre_norm"][full_chunk_batch_idx]

        act_scr = layer.mlp.gate_proj(hidden_states).to(torch.float32).exp()
        denom = torch.zeros(num_batch_idx, seq_len, self.intermediate_size, device=act_scr.device, dtype=torch.float32)
        denom.index_add_(dim=0, index=inverse_chunk_batch_idx.to(act_scr.device), source=act_scr)

        act_weight = (act_scr / denom[inverse_chunk_batch_idx]).to(self.dtype)
        hidden_states = act_weight * cache[f"mlp.{layer_id}.z"][full_chunk_batch_idx]
        hidden_states = layer.mlp.down_proj(hidden_states)

        return residual + hidden_states


################################################################
#                 Causal Patching Baseline                     #
################################################################
    
class APComponentAttributionMethod(ComponentAttributionMethod):

    def __init__(self, model, tokenizer, chunk_size):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size

        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        if not hasattr(model.config, 'head_dim') or model.config.head_dim is None:
            model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size

    def name(self):
        return 'ap'
    
    def get_causal_mask(self, attention_mask):
        batch_size, seq_len = attention_mask.shape

        mask = torch.ones((seq_len, seq_len), device=self.device, dtype=self.dtype)
        mask = torch.tril(mask).bool()
        padding_mask = torch.ones((batch_size, 1, 1, seq_len), device=self.device, dtype=self.dtype)
        padding_mask = padding_mask.masked_fill(attention_mask[:, None, None, :].to(self.device) == 0, 0).bool()
        causal_mask = mask & padding_mask
        return causal_mask.int()
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape

        causal_mask = self.get_causal_mask(batch['attention_mask'])
        attn_mask = causal_mask.repeat(1, self.num_heads, 1, 1)
        attn_abl_idx = attn_mask.nonzero().cpu()
        num_attn_elements = attn_abl_idx.size(0)

        mlp_mask = batch['attention_mask'][:,:, None].repeat(1,1,self.intermediate_size)
        mlp_abl_idx = mlp_mask.nonzero().cpu()
        num_mlp_elements = mlp_abl_idx.size(0)

        attn_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_layers, self.num_heads, dtype=self.dtype)
        mlp_scores = torch.zeros(batch_size, seq_len, self.num_layers, self.intermediate_size, dtype=self.dtype)

        with torch.no_grad():

            with self.model.trace(batch) as tracer:
                hidden_states = self.model.model.embed_tokens.output.save()
                _, kwargs = copy.deepcopy(self.model.model.layers[0].self_attn.inputs)
                attention_mask = kwargs['attention_mask'].save() 
                position_embeddings = kwargs['position_embeddings'].save() 
                tracer.stop()

            for layer_id, layer in enumerate(self.model.model.layers):

                # calculate attn ablations
                for start_idx in range(0, num_attn_elements, self.chunk_size):
                    chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                    chunk_abl_idx = attn_abl_idx[chunk_slice]
                    chunk_batch_idx, head_idx, q_idx, k_idx = chunk_abl_idx.T
                    chunk_len = len(chunk_batch_idx)
                    chunk_mask = attention_mask[chunk_batch_idx]
                    chunk_states = hidden_states[chunk_batch_idx]

                    with layer.trace(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings):
                        layer.self_attn.source.attention_interface_0.source.nn_functional_dropout_0.output[chunk_batch_idx, head_idx, q_idx, k_idx] = 0
                        chunk_states = layer.output.save()
                    
                    for concurrent_layer in self.model.model.layers[layer_id + 1:]:
                        chunk_states = concurrent_layer(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings)
                    
                    logits = self.model.lm_head(self.model.model.norm(chunk_states))[:, -1]
                    abl_probs = torch.softmax(logits, dim=-1)[range(chunk_len), batch['target_ids'][chunk_batch_idx]]

                    attn_scores[chunk_batch_idx, q_idx, k_idx, layer_id, head_idx] = (1 - abl_probs).cpu()
                
                # calculate mlp ablations
                for start_idx in tqdm(range(0, num_mlp_elements, self.chunk_size), desc=f"{layer_id}/{self.num_layers}"):
                    chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                    chunk_abl_idx = mlp_abl_idx[chunk_slice]
                    chunk_batch_idx, s_idx, n_idx= chunk_abl_idx.T
                    chunk_len = len(chunk_batch_idx)
                    chunk_mask = attention_mask[chunk_batch_idx]
                    chunk_states = hidden_states[chunk_batch_idx]

                    with layer.trace(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings):
                        layer.mlp.act_fn.output[chunk_batch_idx, s_idx, n_idx] = 0
                        chunk_states = layer.output.save()
                    
                    for concurrent_layer in self.model.model.layers[layer_id + 1:]:
                        chunk_states = concurrent_layer(hidden_states=chunk_states, attention_mask=chunk_mask, position_embeddings=position_embeddings)
                    
                    logits = self.model.lm_head(self.model.model.norm(chunk_states))[:, -1]
                    abl_probs = torch.softmax(logits, dim=-1)[range(chunk_len), batch['target_ids'][chunk_batch_idx]]

                    mlp_scores[chunk_batch_idx, s_idx, layer_id, n_idx] = (1 - abl_probs).cpu()
                
                hidden_states = layer(hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)

        scores = self._extract_from_mask((attn_scores, mlp_scores), batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
################################################################
#                    Dual Path Attribution                     #
################################################################

class DPAComponentAttribution(ComponentAttributionMethod):
    """
    Wrapper for DPA.
    Allows initializing with different scaling configurations.
    """
    def __init__(self, model, tokenizer, scaling_config):
        super().__init__(model, tokenizer)
        self.scaling_config = scaling_config

        model_id = self.model.config.name_or_path
        if model_id not in BACKEND_MAPPING:
            raise KeyError(f"The model {model_id} is not mapped to a backend.")
        
        self.backend = BACKEND_MAPPING[model_id]
    
    def name(self):
        weight_str = "_".join([str(self.scaling_config[k]).replace(".","") for k in ["q","k", "v", "gate", "up"]])
        return f"dpa_{weight_str}"

    def run(self, batch):
        backend = self.backend(self.model, scaling_config=self.scaling_config, cache_device=self.device)
        tracer = ComponentTracer(backend, self.tokenizer)
        
        tracer_batch = {
            'input_ids': batch['input_ids'].to(self.model.device),
            'attention_mask': batch['attention_mask'].to(self.model.device),
            'targets': batch['target_ids'].to(self.model.device),
        }
        
        # This returns list of lists (scores for each sample)
        scores = tracer.batch_trace(tracer_batch)
        scores = self._extract_from_mask(scores, batch['context_mask'])

        return {
            'ids': batch['ids'],
            'scores': scores
        }