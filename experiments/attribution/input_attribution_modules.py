import torch
import gc
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from abc import ABC, abstractmethod
import copy

from tracer.tracer import InputTracer
from tracer.backend import BACKEND_MAPPING

################################################################
#                       Abstract Classes                       #
################################################################

class InputAttributionMethod(ABC):
    """Abstract base class for all attribution methods."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.dtype = model.dtype

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

    def _extract_context(self, flat_tensor, batch):
        """Helper to extract context-only tokens from flat tensors based on masks."""
        # This handles the logic of unflattening the nnsight output
        lengths = batch['context_mask'].sum(dim=1).tolist()
        flat_context_data = flat_tensor[batch['context_mask'].bool()]
        return [t.cpu() for t in torch.split(flat_context_data, lengths)]

    def _extract_full(self, flat_tensor, batch):
        """Helper to extract all tokens."""
        lengths = batch['attention_mask'].sum(dim=1).tolist()
        flat_data = flat_tensor[batch['attention_mask'].bool()]
        return [t.tolist() for t in torch.split(flat_data, lengths)]


################################################################
#                       Random Baseline                        #
################################################################

class RandomInputAttribution(InputAttributionMethod):

    def name(self):
        return "random"

    def run(self, batch):

        # Calculate context lengths to generate random noise of correct shape
        context_lengths = batch['context_mask'].sum(dim=1).tolist()
        scores = [torch.randn(length) for length in context_lengths]

        return {
            'ids': batch['ids'],
            'scores': scores
        }


################################################################
#                     Attention Baselines                      #
################################################################


class LastLayerAttentionInputAttribution(InputAttributionMethod):

    def name(self):
        return "last_layer_attn"

    def run(self, batch):
        
        batch_attn = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:
                batch_attn.append(
                    layer.self_attn.output[1].save()
                )
            
        # (Layers, Batch, Heads, Q, K)
        batch_attn = torch.stack([t.to(self.device) for t in batch_attn])
        
        last_layer_attn = batch_attn.mean(dim=2)[-1, :, -1, :] # (B, K)
        scores = self._extract_context(last_layer_attn, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }

class MeanAttentionInputAttribution(InputAttributionMethod):

    def name(self):
        return "mean_attn"

    def run(self, batch):
        
        batch_attn = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:
                batch_attn.append(
                    layer.self_attn.output[1].save()
                )
            
        # (Layers, Batch, Heads, Q, K)
        batch_attn = torch.stack([t.to(self.device) for t in batch_attn])

        mean_attn = batch_attn.mean(dim=(0, 2))[:, -1, :] # (B, K)
        scores = self._extract_context(mean_attn, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }

class AttentionRolloutInputAttribution(InputAttributionMethod):

    def name(self):
        return "attn_rollout"

    def run(self, batch):
        seq_len = batch['input_ids'].shape[1]
        
        batch_attn = []
        with torch.no_grad(), self.model.trace(batch):
            for layer in self.model.model.layers:
                batch_attn.append(
                    layer.self_attn.output[1].save()
                )
            
        # (Layers, Batch, Heads, Q, K)
        batch_attn = torch.stack([t.to(self.device) for t in batch_attn])

        residualized_attentions = (0.5 * batch_attn.mean(dim=2) + 0.5 * torch.eye(seq_len)[None].to(device=self.model.device, dtype=self.model.dtype))
        rollout = residualized_attentions[0]
        for layer in range(1, self.model.config.num_hidden_layers):
            rollout = torch.matmul(residualized_attentions[layer], rollout)
        scores = self._extract_context(rollout[:, -1], batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
################################################################
#                     Gradient Baselines                       #
################################################################

class GradientInputAttribution(InputAttributionMethod):

    def name(self):
        return "gradient"

    def run(self, batch):
        curr_bs = batch['input_ids'].size(0)

        with self.model.trace(batch):
            emb = self.model.model.embed_tokens.output

            logits = self.model.lm_head.output[:, -1]
            target_logits = logits[range(curr_bs), batch['target_ids']]
            grads = torch.autograd.grad(
                outputs=target_logits, 
                inputs=emb,
                grad_outputs=torch.ones_like(target_logits)
            )[0].save()


        gradient = torch.sum(grads.to(self.device), dim=-1)
        scores = self._extract_context(gradient, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
class InputXGradientInputAttribution(InputAttributionMethod):

    def name(self):
        return "input_x_gradient"

    def run(self, batch):
        curr_bs = batch['input_ids'].size(0)

        with self.model.trace(batch):
            emb = self.model.model.embed_tokens.output.save()

            logits = self.model.lm_head.output[:, -1]
            target_logits = logits[range(curr_bs), batch['target_ids']]
            grads = torch.autograd.grad(
                outputs=target_logits, 
                inputs=emb,
                grad_outputs=torch.ones_like(target_logits)
            )[0].save()

        gradient = torch.sum(emb.to(self.device) * grads.to(self.device), dim=-1)
        scores = self._extract_context(gradient, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }


class IntegratedGradientInputAttribution(InputAttributionMethod):
    def __init__(self, model, tokenizer, steps=100): # Defaults to 20 for speed, use 100 for final
        super().__init__(model, tokenizer)
        self.steps = steps

    def name(self):
        return "integrated_gradient"

    def run(self, batch):
        curr_bs = batch['input_ids'].size(0)
        all_grads = []

        # 1. Get original embeddings first (to scale against)
        with self.model.trace(batch):
            original_embeds_val = self.model.model.embed_tokens.output.save()

        # 2. Path Integral Loop
        for i in range(1, self.steps + 1):
            alpha = i / self.steps

            with self.model.trace(batch):
                scaled_embeds = (original_embeds_val * alpha).detach().requires_grad_(True)
                self.model.model.embed_tokens.output = scaled_embeds

                logits = self.model.lm_head.output[:, -1]
                target_logits = logits[range(curr_bs), batch['target_ids']]

                grads = torch.autograd.grad(
                    outputs=target_logits, 
                    inputs=scaled_embeds,
                    grad_outputs=torch.ones_like(target_logits)
                )[0].save()
            
            all_grads.append(grads)

        avg_grads = torch.stack([t.to(self.device) for t in all_grads]).sum(dim=0) / self.steps
        ig_scores = (avg_grads * original_embeds_val.to(self.device)).sum(dim=-1)
        scores = self._extract_context(ig_scores, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }
    
################################################################
#                  Decomposition Baselines                     #
################################################################

class DePassInputAttribution(InputAttributionMethod):
    """Wrapper for DePass attribution."""

    def __init__(self, model, tokenizer, chunk_size = 64):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size

        self.num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
        if not hasattr(model.config, 'head_dim') or model.config.head_dim is None:
            model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size

    def name(self):
        return "depass"
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape
        ablation_idx = batch['context_mask'].nonzero()
        num_elements = ablation_idx.size(0)

        scores = torch.zeros(batch_size, seq_len, dtype=self.dtype)

        with torch.no_grad():

            cache = {}
            with self.model.trace(batch) as tracer:
                embeddings = self.model.model.embed_tokens.output.save()

                for layer_id, layer in enumerate(self.model.model.layers):

                    cache[f"attn.{layer_id}.pre_norm"] = self.get_rmsnorm_scaling(layer.input_layernorm.input).save()
                    cache[f"attn.{layer_id}.weights"] = layer.self_attn.output[1].save()

                    cache[f"mlp.{layer_id}.pre_norm"] = self.get_rmsnorm_scaling(layer.post_attention_layernorm.input).save()
                    cache[f"mlp.{layer_id}.z"] = layer.mlp.down_proj.input.save()
                
                norm_scalar = self.get_rmsnorm_scaling(self.model.model.norm.input).save()

            for start_idx in range(0, num_elements, self.chunk_size):
                chunk_slice = slice(start_idx, start_idx + self.chunk_size)
                chunk_batch_idx, token_idx = ablation_idx[chunk_slice].T
                num_chunk_elements = len(chunk_batch_idx)

                ablation_mask = torch.zeros(num_chunk_elements, seq_len)
                ablation_mask[range(num_chunk_elements), token_idx] = True\
                
                residual_batch_idx = torch.unique(chunk_batch_idx)
                num_batch_idx = len(residual_batch_idx)
                residual_mask = batch['attention_mask'].clone()
                residual_mask[chunk_batch_idx, token_idx] = 0
                residual_mask = residual_mask[residual_batch_idx]
                batch_idx_needs_residual = residual_mask.any(-1)

                residual_batch_idx = residual_batch_idx[batch_idx_needs_residual]
                residual_mask = residual_mask[batch_idx_needs_residual]

                full_mask = ~torch.concat([ablation_mask, residual_mask]).bool() # concat and invert mask
                full_chunk_batch_idx = torch.concat([chunk_batch_idx, residual_batch_idx])
                inverse_chunk_batch_idx = torch.unique(full_chunk_batch_idx, return_inverse=True)[1]

                chunk_hs = embeddings[full_chunk_batch_idx]
                chunk_hs[full_mask] = 0

                for layer_id, layer in enumerate(self.model.model.layers):
                    chunk_hs = self.process_decomposed_attn(layer, layer_id, chunk_hs, cache, full_chunk_batch_idx)

                    chunk_hs = self.process_decomposed_mlp(layer, layer_id, chunk_hs, cache, full_chunk_batch_idx, inverse_chunk_batch_idx, num_batch_idx)

                chunk_hs = chunk_hs * self.model.model.norm.weight.data * norm_scalar[full_chunk_batch_idx]
                logits = self.model.lm_head(chunk_hs)[range(num_chunk_elements), -1, batch['target_ids'][chunk_batch_idx].tolist()]
                
                scores[chunk_batch_idx, token_idx] = logits.cpu()

        scores = self._extract_context(scores, batch)
 
        return {
            'ids': batch['ids'],
            'scores': scores
        }

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
        hidden_states = layer.self_attn.o_proj(hidden_states)
        
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

class CompleteDePassInputAttribution(InputAttributionMethod):
    """Alternative depass implementation that ensures completenes."""
    
    def name(self):
        return "complete_depass"

    def run(self, batch):

        with torch.no_grad(), self.model.trace(batch) as tracer:
            hidden_states = self.model.model.embed_tokens.output.save()
            args, kwargs = copy.deepcopy(self.model.model.layers[0].self_attn.inputs)
            attention_mask = kwargs['attention_mask'].save() 
            position_embeddings = kwargs['position_embeddings'].save() 
            tracer.stop()

        batch_size, seq_len = batch['input_ids'].shape
        batch_idx, token_idx = batch['context_mask'].nonzero(as_tuple=True)
        num_elements = batch_idx.size(0)

        decomposed_states = torch.zeros(num_elements, seq_len, self.model.config.hidden_size, dtype=self.model.dtype)
        decomposed_states[range(num_elements), token_idx] = hidden_states[batch_idx, token_idx].cpu()

        # add residual as catch-it-all
        residual = hidden_states.cpu()
        residual[batch['context_mask'].bool()] = 0
        decomposed_states = torch.concat([decomposed_states, residual], dim=0)
        batch_idx = torch.concat([batch_idx, torch.arange(batch_size)])
        num_elements = num_elements + batch_size

        with torch.no_grad():

            for layer in self.model.model.layers:

                hidden_states, decomposed_states = self.process_attn(layer, hidden_states, decomposed_states, attention_mask, position_embeddings, batch_idx, batch_size, num_elements)
                
                hidden_states, decomposed_states = self.process_mlp(layer, hidden_states, decomposed_states, batch_idx, batch_size, seq_len, num_elements)

        flat_scores = []
        for start_idx in range(0, num_elements, batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            curr_batch_idx = batch_idx[batch_slice]

            batch_scores = self.model.lm_head(self.model.model.norm(decomposed_states[batch_slice].to(self.model.device)))[range(len(curr_batch_idx)), -1, batch['target_ids'][curr_batch_idx]]
            flat_scores.extend(batch_scores)

        flat_scores = torch.stack(flat_scores)[:-batch_size] # remove residuals
        length = batch['context_mask'].sum(-1)
        scores = [t.cpu() for t in torch.split(flat_scores, length.tolist())]

        return {
            'ids': batch['ids'],
            'scores': scores
        }

    def get_rmsnorm_scaling(
        self,
        residual_state: torch.Tensor
    ) -> torch.Tensor:
        # transfer to float32 to avoid rounding errors during exponentiation
        input_dtype = residual_state.dtype
        residual_state = residual_state.to(torch.float32)

        variance = residual_state.pow(2).mean(-1, keepdim=True)
        rmsnorm_salar = torch.rsqrt(variance + 1e-7) # 1 / sqrt(x)
        return rmsnorm_salar.to(input_dtype)

    def process_attn(
            self, 
            layer, 
            hidden_states, 
            decomposed_states, 
            attention_mask,
            position_embeddings,
            batch_idx, 
            batch_size, 
            num_elements
        ):
        residual = hidden_states

        # get original norm and attn activations
        norm_scalar = self.get_rmsnorm_scaling(hidden_states)
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, attn_weights = layer.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states.to(residual.device)

        # apply norm and attn activation to decomposed sate
        for start_idx in range(0, num_elements, batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            curr_batch_idx = batch_idx[batch_slice]

            # apply pre attn norm
            batch_states = decomposed_states[batch_slice].to(self.model.device)
            batch_states *= norm_scalar[curr_batch_idx].to(self.model.device) * layer.input_layernorm.weight.data.to(self.model.device)
            
            with layer.self_attn.trace(**{'hidden_states': batch_states, 'attention_mask': attention_mask[curr_batch_idx], 'position_embeddings': position_embeddings}):
                layer.self_attn.source.attention_interface_0.source.nn_functional_dropout_0.output = attn_weights[curr_batch_idx]
                batch_states = layer.self_attn.output[0].save()

            decomposed_states[batch_slice] += batch_states.cpu()
        
        return hidden_states, decomposed_states

    def process_mlp(self, layer, hidden_states, decomposed_states, batch_idx, batch_size, seq_len, num_elements):
            
        residual = hidden_states

        # get original norm and mlp activations
        norm_scalar = self.get_rmsnorm_scaling(hidden_states)
        hidden_states = layer.post_attention_layernorm(hidden_states)
        with layer.mlp.trace(hidden_states):
            mlp_act = layer.mlp.act_fn.output.save()
            mlp_up = layer.mlp.up_proj.output.save()

            hidden_states = layer.mlp.output.save()
        hidden_states = residual + hidden_states.to(residual.device)

        act_scores = torch.zeros(batch_size, seq_len, self.model.config.intermediate_size, dtype=torch.float32, device=self.model.device)
        decomposed_scr_cpu = torch.zeros(num_elements, seq_len, self.model.config.intermediate_size, dtype=self.model.dtype)
        
        for start_idx in range(0, num_elements, batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            curr_batch_idx = batch_idx[batch_slice]

            batch_states = decomposed_states[batch_slice].to(self.model.device)
            batch_states *= norm_scalar[curr_batch_idx].to(self.model.device) * layer.post_attention_layernorm.weight.data.to(self.model.device)

            with layer.mlp.trace(batch_states):
                mlp_scr = layer.mlp.act_fn.input.save() 
            
            decomposed_scr_cpu[batch_slice] = mlp_scr.cpu()
            act_scores.index_add_(dim=0, index=curr_batch_idx.to(self.model.device), source=mlp_scr.to(torch.float32).exp().to(self.model.device))
            
        for start_idx in range(0, num_elements, batch_size):
            batch_slice = slice(start_idx, start_idx + batch_size)
            curr_batch_idx = batch_idx[batch_slice]

            chunk_scr = decomposed_scr_cpu[batch_slice].to(device=self.model.device, dtype=torch.float32).exp()
            chunk_weights = (chunk_scr / act_scores[curr_batch_idx]).to(self.model.dtype) 

            chunk_output = layer.mlp.down_proj(chunk_weights.to(dtype=self.model.dtype, device=self.model.device) * mlp_act[curr_batch_idx].to(dtype=self.model.dtype, device=self.model.device) * mlp_up[curr_batch_idx].to(dtype=self.model.dtype, device=self.model.device))
            
            decomposed_states[batch_slice] += chunk_output.cpu()
        
        return hidden_states, decomposed_states

    
class IFRInputAttribution(InputAttributionMethod):

    def __init__(self, model, tokenizer, chunk_size):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size

        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_attention_heads = self.model.config.num_attention_heads
        self.num_key_value_heads= self.model.config.num_key_value_heads
        if not hasattr(model.config, 'head_dim') or model.config.head_dim is None:
            model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = model.config.head_dim

        self.num_head_groups = self.num_attention_heads // self.num_key_value_heads

    def name(self):
        return 'ifr'
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape
        context_mask_idx = batch['attention_mask'].nonzero()
        batch_idx, token_idx = context_mask_idx.T
        num_elements = len(batch_idx)

        attn_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_hidden_layers, self.num_attention_heads, dtype=self.dtype)

        with torch.no_grad(), self.model.trace(batch):
            for layer_id, layer in enumerate(self.model.model.layers):

                # Attn Norms
                v_proj = layer.self_attn.v_proj.output
                attn_out, attn_weight = layer.self_attn.output
                o_proj_WT = layer.self_attn.o_proj.weight.data.T

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
                    attn_scores[chunk_batch_idx, token_idx, :, layer_id] = layer_attn_scores.save().cpu()

            logits = self.model.lm_head.output[range(batch_size), -1, batch['target_ids']].save()

        grad = torch.zeros(batch_size, seq_len, device=logits.device, dtype=logits.dtype)
        grad[:, -1] = logits

        for layer_id in range(self.num_hidden_layers - 1, -1, -1):
            attn_scores[:, :, :, layer_id] = (attn_scores[:, :, :, layer_id].to(grad.device) * grad[:, :, None, None]).cpu()

            grad = attn_scores[:, :, :, layer_id].sum((-3, -1))

        scores = self._extract_context(grad, batch)
 
        return {
            'ids': batch['ids'],
            'scores': scores
        }

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
            
        if batch_token_index:
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
        if batch_token_index:
            batch_idx, token_idx = batch_token_index
            act_prod = act_prod[batch_idx, token_idx]

            decomposed_glu = torch.einsum('bh,hm->bhm', act_prod, down_proj_WT)

            return decomposed_glu
        
        decomposed_glu = torch.einsum('bth,hm->bthm', act_prod, down_proj_WT)
        
        return decomposed_glu

################################################################
#                 Causal Patching Baseline                     #
################################################################   

class APInputAttribution(InputAttributionMethod):

    def __init__(self, model, tokenizer, chunk_size):
        super().__init__(model, tokenizer)
        self.chunk_size = chunk_size

    def name(self):
        return 'ap'
    
    def run(self, batch):
        batch_size, seq_len = batch['input_ids'].shape
        ablation_idx = batch['context_mask'].nonzero()
        num_elements = ablation_idx.size(0)

        flat_scores = []
        for start_idx in range(0, num_elements, self.chunk_size):
            batch_slice = slice(start_idx, start_idx + self.chunk_size)
            batch_idx, token_idx = ablation_idx[batch_slice].T
            with torch.no_grad(), self.model.trace({'input_ids': batch['input_ids'][batch_idx], 'attention_mask': batch['attention_mask'][batch_idx]}):
                self.model.model.embed_tokens.output[range(len(batch_idx)), token_idx]= 0
                logits = self.model.lm_head.output[range(len(batch_idx)), -1].save()
                probs = torch.softmax(logits, dim=-1)[range(len(batch_idx)), batch['target_ids'][batch_idx]]
                flat_scores.extend(1 - probs)
            
        flat_scores = torch.stack(flat_scores)
        length = batch['context_mask'].sum(-1)
        scores = [t.cpu() for t in torch.split(flat_scores, length.tolist())]
        return {
            'ids': batch['ids'],
            'scores': scores
        }


################################################################
#                    Dual Path Attribution                     #
################################################################

class DPAInputAttribution(InputAttributionMethod):
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
        tracer = InputTracer(backend, self.tokenizer)
        
        tracer_batch = {
            'input_ids': batch['input_ids'].to(self.model.device),
            'attention_mask': batch['attention_mask'].to(self.model.device),
            'targets': batch['target_ids'].to(self.model.device),
        }
        
        # This returns list of lists (scores for each sample)
        scores = tracer.batch_trace(tracer_batch)[0]
        scores = self._extract_context(scores, batch)

        return {
            'ids': batch['ids'],
            'scores': scores
        }