from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
from torch import nn
from nnsight import LanguageModel

from tracer.utils import (
    get_causal_mask,
    get_sliding_window_causal_mask,
    get_rmsnorm_scaling,
    rotate_half,
    decompose_attention_to_head,
    decompose_glu_to_neuron,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, 
)

class ModelBackend(ABC):
    """
    Abstract base class for defining model-specific tracing backends.
    """

    def __init__(
        self,
        model: LanguageModel,
        scaling_config: Dict[str, float] = None,
        cache_device: torch.device | str | None = None,
    ):
        """
        Initializes the model backend.

        Args:
            model (LanguageModel): The nnsight language model to trace.
        """
        self.model = model
        self.config = model.config
        self.compute_device = model.device
        self.dtype = model.dtype

        self.num_head_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        self.scaling =  scaling_config or {
            'up': 0.5, 'gate': 0.5, 'v': 0.5, 'q': 0.25, 'k': 0.25
        }
        self.cache_device = torch.device(cache_device) if cache_device is not None else self.compute_device

    def _move_cache_to_cache_device(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        if self.cache_device == self.compute_device:
            return cache
        for key, value in cache.items():
            if torch.is_tensor(value) and value.device != self.cache_device:
                cache[key] = value.to(self.cache_device)
        return cache

    @abstractmethod
    def run_forward_and_cache(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the nnsight trace context and caches necessary activations.

        Args:
            batch (Dict[str, Any]): The input batch dictionary containing input_ids and attention_mask.

        Returns:
            Dict[str, Any]: A dictionary containing all necessary cached activations and proxies.
        """
        pass

    @abstractmethod
    def get_mlp_update(self, layer_idx: int, current_grad: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        """
        Calculates the gradient contribution from the MLP block at a specific layer.

        Args:
            layer_idx (int): The index of the layer to compute the update for.
            current_grad (torch.Tensor): The current gradient tensor in the residual stream.
            cache (Dict[str, Any]): The dictionary of cached activations.

        Returns:
            torch.Tensor: The update vector to be added to the residual stream.
        """
        pass

    @abstractmethod
    def get_attn_update(self, layer_idx: int, current_grad: torch.Tensor, cache: Dict[str, Any]) -> torch.Tensor:
        """
        Calculates the gradient contribution from the Attention block at a specific layer.

        Args:
            layer_idx (int): The index of the layer to compute the update for.
            current_grad (torch.Tensor): The current gradient tensor in the residual stream.
            cache (Dict[str, Any]): The dictionary of cached activations.

        Returns:
            torch.Tensor: The update vector to be added to the residual stream.
        """
        pass

    @abstractmethod
    def get_final_norm_scale(self, cache: Dict[str, Any]) -> torch.Tensor:
        """
        Retrieves the final LayerNorm scaling factor from the activation cache.

        Args:
            cache (Dict[str, Any]): The dictionary of cached activations.

        Returns:
            torch.Tensor: The scaling factor tensor from the final layer normalization.
        """
        pass

    @abstractmethod
    def get_component_contributions(self, batch: Dict[str, Any], residual_targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the layer-wise component contributions for attention and MLP blocks.

        Args:
            batch (Dict[str, Any]): The input batch dictionary.
            residual_targets (Dict[str, torch.Tensor]): The dictionary mapping layer targets to their respective residual states.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the attention and MLP contribution tensors.
        """
        pass

class Llama2Backend(ModelBackend):
    
    def run_forward_and_cache(self, batch: Dict) -> Dict[str, Any]:
        curr_batch_size, seq_len = batch['input_ids'].shape
        cache = {}

        # # TODO remove when fixed
        # with self.model.trace("Paris is in"):
        #     for layer in self.model.model.layers:
        #         layer.self_attn.source.apply_rotary_pos_emb_0.output

        with torch.no_grad(), self.model.trace(batch):
            # Embeddings
            cache['emb'] = self.model.model.embed_tokens.output.save()
            
            # RoPE
            cos, sin = self.model.model.rotary_emb.output
            cache['rotary.cos'] = cos.save()
            cache['rotary.sin'] = sin.save()

            for i, layer in enumerate(self.model.model.layers):
                # Input Norm
                cache[f'ln.attn.{i}'] = get_rmsnorm_scaling(layer.input).save() # (B, Q, 1)

                # Attention 
                v_proj = layer.self_attn.source.self_v_proj_0.output.view(curr_batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
                rot_q_proj, rot_k_proj = layer.self_attn.source.apply_rotary_pos_emb_0.output.save()
                if self.num_head_groups > 1:
                    v_proj = v_proj.repeat_interleave(self.num_head_groups, dim=1)
                    rot_k_proj = rot_k_proj.repeat_interleave(self.num_head_groups, dim=1)
                cache[f"attn.{i}.v"] = v_proj.save() # (B, H, K, D)
                cache[f"attn.{i}.rot_q"] = rot_q_proj.save() # (B, H, Q, D)
                cache[f"attn.{i}.rot_k"] = rot_k_proj.save() # (B, H, K, D)

                cache[f"attn.{i}.weights"] = layer.self_attn.source.attention_interface_0.output[1].save() # (B, H, Q, K)
                cache[f"attn.{i}.z"] = layer.self_attn.source.self_o_proj_0.input.view(curr_batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2).save() # (B, H, Q, D)

                # Post Attn Norm
                cache[f'ln.mlp.{i}'] = get_rmsnorm_scaling(layer.post_attention_layernorm.input).save()

                # MLP 
                cache[f"mlp.{i}.gate_proj"] = layer.mlp.gate_proj.output.save() # (B, Q, D)
                cache[f"mlp.{i}.gate_act"] = layer.mlp.act_fn.output.save() # (B, Q, D)
                cache[f"mlp.{i}.up_proj"] = layer.mlp.up_proj.output.save() # (B, Q, D)

            cache['final_ln'] = get_rmsnorm_scaling(self.model.model.norm.input[:, -1]).save() # (B, Q, 1)
        
        return self._move_cache_to_cache_device(cache)
    
    def get_mlp_update(self, layer_idx: int, grad: torch.Tensor, cache: Dict) -> torch.Tensor:
        layer = self.model.model.layers[layer_idx]
        norm_w = layer.post_attention_layernorm.weight.data.to(self.compute_device)
        up_w = layer.mlp.up_proj.weight.data.to(self.compute_device)
        gate_w = layer.mlp.gate_proj.weight.data.to(self.compute_device)
        down_w = layer.mlp.down_proj.weight.data.to(self.compute_device)

        gate_proj = cache[f'mlp.{layer_idx}.gate_proj'].to(self.compute_device)
        up_proj = cache[f'mlp.{layer_idx}.up_proj'].to(self.compute_device)
        gate_act = cache[f'mlp.{layer_idx}.gate_act'].to(self.compute_device)
        ln_scalar = cache[f'ln.mlp.{layer_idx}'].to(self.compute_device)

        # Grad Mid Projection
        grad_mid = torch.matmul(grad.to(self.compute_device), down_w) 

        # Math: Up Path
        eff_w_up = up_w * norm_w
        glu_scalar_up = gate_act * ln_scalar
        up_term = torch.matmul(grad_mid * glu_scalar_up, eff_w_up)

        # Math: Gate Path
        eff_w_gate = gate_w * norm_w
        glu_scalar_gate = gate_proj.sigmoid() * up_proj * ln_scalar
        gate_term = torch.matmul(grad_mid * glu_scalar_gate, eff_w_gate)

        return self.scaling['up'] * up_term + self.scaling['gate'] * gate_term

    def get_attn_update(self, layer_idx: int, grad: torch.Tensor, cache: Dict) -> torch.Tensor:
        batch_size, seq_len = grad.shape[:2]
        layer = self.model.model.layers[layer_idx]
        norm_w = layer.input_layernorm.weight.data.to(self.compute_device)
        q_w = layer.self_attn.q_proj.weight.data.view(self.config.num_attention_heads, self.config.head_dim, self.config.hidden_size).to(self.compute_device)
        k_w = layer.self_attn.k_proj.weight.data.view(self.config.num_key_value_heads, self.config.head_dim, self.config.hidden_size).to(self.compute_device)
        v_w = layer.self_attn.v_proj.weight.data.view(self.config.num_key_value_heads, self.config.head_dim, self.config.hidden_size).to(self.compute_device)
        if self.num_head_groups > 1:
            v_w = v_w.repeat_interleave(self.num_head_groups, 0)
            k_w = k_w.repeat_interleave(self.num_head_groups, 0)
        o_w = layer.self_attn.o_proj.weight.data.to(self.compute_device)

        sin = cache[f"rotary.sin"].to(self.compute_device)
        cos = cache[f"rotary.cos"].to(self.compute_device)
        v_proj = cache[f"attn.{layer_idx}.v"].to(self.compute_device)
        rot_q_proj = cache[f"attn.{layer_idx}.rot_q"].to(self.compute_device)
        rot_k_proj = cache[f"attn.{layer_idx}.rot_k"].to(self.compute_device)
        attn_weight = cache[f"attn.{layer_idx}.weights"].to(self.compute_device)
        z = cache[f"attn.{layer_idx}.z"].to(self.compute_device)
        ln_scalar = cache[f"ln.attn.{layer_idx}"].to(self.compute_device)

        # Grad Mid Projection
        grad_mid = torch.matmul(grad.to(self.compute_device), o_w).view(batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2) # (B, H, Q, D)

        # Math: Value Path
        eff_w_v = v_w * norm_w
        v_scalar = (attn_weight * ln_scalar.transpose(-1,-2).unsqueeze(1)).transpose(-2, -1) 
        grad_t = torch.matmul(v_scalar, grad_mid).transpose(1,2)
        v_term = torch.matmul(grad_t.flatten(-2), eff_w_v.flatten(0, 1))

        # 
        term_1 = torch.matmul(grad_mid, v_proj.transpose(-1, -2))
        term_2 = (grad_mid * z).sum(dim=-1, keepdim=True)
        delta_ij = attn_weight * (term_1 - term_2) # Shape (B, H, Q, K)

        # Math Query Pass
        weighted_k_sum = torch.matmul(delta_ij, rot_k_proj) * (self.config.head_dim ** -0.5)
        grad_q_rot = (weighted_k_sum * cos) - (rotate_half(weighted_k_sum) * sin) # + for inverse if sin was - forward
        eff_q_w = (q_w * norm_w).flatten(0, 1)
        q_term = torch.matmul(grad_q_rot.transpose(1, 2).flatten(-2), eff_q_w) * ln_scalar

        weighted_q_sum = torch.matmul(delta_ij.transpose(-2, -1), rot_q_proj) * (self.config.head_dim ** -0.5)
        grad_k_rot = (weighted_q_sum * cos) - (rotate_half(weighted_q_sum) * sin) # + for inverse if sin was - forward
        eff_k_w = (k_w * norm_w).flatten(0, 1)
        k_term = torch.matmul(grad_k_rot.transpose(1, 2).flatten(-2), eff_k_w) * ln_scalar

        return (self.scaling['v'] * v_term + 
                self.scaling['q'] * q_term + 
                self.scaling['k'] * k_term)

    def get_final_norm_scale(self, cache: Dict):
        return cache['final_ln']
    
    def get_component_contributions(self, batch: Dict[str, Any], residual_targets: Dict[str, torch.Tensor]):
        batch_attn = []
        batch_mlp = []
        with torch.no_grad(), self.model.trace(batch):
            for layer_id, layer in enumerate(self.model.model.layers):
                v_proj = layer.self_attn.v_proj.output
                attn_weight = layer.self_attn.output[1]
                o_proj_WT = layer.self_attn.o_proj.weight.data.T
                d_attn = decompose_attention_to_head(
                    attn_weight,
                    v_proj,
                    o_proj_WT,
                    num_attention_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                ) # (B, Q, K, H, D)
                batch_attn.append(
                    torch.einsum('bqd, bq...d -> bq...', residual_targets[f"mid.{layer_id}"].to(self.compute_device), d_attn.to(self.compute_device)).save()# (B, Q, K, H)
                )
                del d_attn

                act_prod = layer.mlp.down_proj.input
                down_proj_WT = layer.mlp.down_proj.weight.data.T
                d_mlp = decompose_glu_to_neuron(
                    down_proj_WT,
                    act_prod
                ) # (B, Q, I, D)
                batch_mlp.append(
                    torch.einsum('bqd, bq...d -> bq...', residual_targets[f"mid.{layer_id}"].to(self.compute_device), d_mlp.to(self.compute_device)).save() # (B, Q, I)
                )
                del d_mlp
        batch_attn = torch.stack(batch_attn).permute(1, 2, 3, 0, 4).to(self.cache_device) # (B, Q, K, L, H)
        batch_mlp = torch.stack(batch_mlp).permute(1, 2, 0, 3).to(self.cache_device) # (B, Q, L, I)
        return batch_attn, batch_mlp
    

class Qwen3Backend(ModelBackend):

    def __init__(self, model, scaling_config = None, cache_device = None):
        super().__init__(model, scaling_config, cache_device)
        if model.config.use_sliding_window:
            raise ValueError("Please deactivate sliding window with 'use_sliding_window=False'.")

    def run_forward_and_cache(self, batch: Dict) -> Dict[str, Any]:
        curr_batch_size, seq_len = batch['input_ids'].shape
        cache = {}

        with torch.no_grad(), self.model.trace(batch):
            # Embeddings
            cache['emb'] = self.model.model.embed_tokens.output.save()
            
            # RoPE
            cos, sin = self.model.model.rotary_emb.output
            cache['rotary.cos'] = cos.save()
            cache['rotary.sin'] = sin.save()

            for i, layer in enumerate(self.model.model.layers):
                # Input Norm
                cache[f'ln.attn.{i}'] = get_rmsnorm_scaling(layer.input).save() # (B, Q, 1)

                # Attention 
                cache[f'attn.{i}.q_norm'] = get_rmsnorm_scaling(layer.self_attn.source.view_0.output).transpose(1, 2).save() # (B, H, Q, 1) diff to llama q_norm
                k_norm = get_rmsnorm_scaling(layer.self_attn.source.view_1.output).transpose(1, 2) 
                v_proj = layer.self_attn.source.self_v_proj_0.output.view(curr_batch_size, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
                rot_q_proj, rot_k_proj = layer.self_attn.source.apply_rotary_pos_emb_0.output.save()
                if self.num_head_groups > 1:
                    k_norm = k_norm.repeat_interleave(self.num_head_groups, dim=1)
                    v_proj = v_proj.repeat_interleave(self.num_head_groups, dim=1)
                    rot_k_proj = rot_k_proj.repeat_interleave(self.num_head_groups, dim=1)
                cache[f'attn.{i}.k_norm'] = k_norm.save() # (B, H, K, 1) diff to llama k_norm
                cache[f"attn.{i}.v"] = v_proj.save() # (B, H, K, D)
                cache[f"attn.{i}.rot_q"] = rot_q_proj.save() # (B, H, Q, D)
                cache[f"attn.{i}.rot_k"] = rot_k_proj.save() # (B, H, K, D)

                cache[f"attn.{i}.weights"] = layer.self_attn.source.attention_interface_0.output[1].save() # (B, H, Q, K)
                cache[f"attn.{i}.z"] = layer.self_attn.source.self_o_proj_0.input.view(curr_batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2).save() # (B, H, Q, D)

                # Post Attn Norm
                cache[f'ln.mlp.{i}'] = get_rmsnorm_scaling(layer.post_attention_layernorm.input).save()

                # MLP 
                cache[f"mlp.{i}.gate_proj"] = layer.mlp.gate_proj.output.save() # (B, Q, D)
                cache[f"mlp.{i}.gate_act"] = layer.mlp.act_fn.output.save() # (B, Q, D)
                cache[f"mlp.{i}.up_proj"] = layer.mlp.up_proj.output.save() # (B, Q, D)

            cache['final_ln'] = get_rmsnorm_scaling(self.model.model.norm.input[:, -1]).save() # (B, Q, 1)
        
        return self._move_cache_to_cache_device(cache)
    
    def get_mlp_update(self, layer_idx: int, grad: torch.Tensor, cache: Dict) -> torch.Tensor:
        layer = self.model.model.layers[layer_idx]
        norm_w = layer.post_attention_layernorm.weight.data
        up_w = layer.mlp.up_proj.weight.data
        gate_w = layer.mlp.gate_proj.weight.data
        down_w = layer.mlp.down_proj.weight.data

        gate_proj = cache[f'mlp.{layer_idx}.gate_proj'].to(self.compute_device)
        up_proj = cache[f'mlp.{layer_idx}.up_proj'].to(self.compute_device)
        gate_act = cache[f'mlp.{layer_idx}.gate_act'].to(self.compute_device)
        ln_scalar = cache[f'ln.mlp.{layer_idx}'].to(self.compute_device)

        # Grad Mid Projection
        grad_mid = torch.matmul(grad.to(self.compute_device), down_w) 

        # Math: Up Path
        eff_w_up = up_w * norm_w
        glu_scalar_up = gate_act * ln_scalar
        up_term = torch.matmul(grad_mid * glu_scalar_up, eff_w_up)

        # Math: Gate Path
        eff_w_gate = gate_w * norm_w
        glu_scalar_gate = gate_proj.sigmoid() * up_proj * ln_scalar
        gate_term = torch.matmul(grad_mid * glu_scalar_gate, eff_w_gate)

        return self.scaling['up'] * up_term + self.scaling['gate'] * gate_term

    def get_attn_update(self, layer_idx: int, grad: torch.Tensor, cache: Dict) -> torch.Tensor:
        batch_size, seq_len = grad.shape[:2]
        layer = self.model.model.layers[layer_idx]
        norm_w = layer.input_layernorm.weight.data
        qn_w = layer.self_attn.q_norm.weight.data # diff to llama -> qn_w
        kn_w = layer.self_attn.k_norm.weight.data # diff to llama -> kn_w
        q_w = layer.self_attn.q_proj.weight.data.view(self.config.num_attention_heads, self.config.head_dim, self.config.hidden_size)
        k_w = layer.self_attn.k_proj.weight.data.view(self.config.num_key_value_heads, self.config.head_dim, self.config.hidden_size)
        v_w = layer.self_attn.v_proj.weight.data.view(self.config.num_key_value_heads, self.config.head_dim, self.config.hidden_size)
        if self.num_head_groups > 1:
            v_w = v_w.repeat_interleave(self.num_head_groups, 0)
            k_w = k_w.repeat_interleave(self.num_head_groups, 0)
        o_w = layer.self_attn.o_proj.weight.data

        sin = cache[f"rotary.sin"].to(self.compute_device)
        cos = cache[f"rotary.cos"].to(self.compute_device)
        v_proj = cache[f"attn.{layer_idx}.v"].to(self.compute_device)
        rot_q_proj = cache[f"attn.{layer_idx}.rot_q"].to(self.compute_device)
        rot_k_proj = cache[f"attn.{layer_idx}.rot_k"].to(self.compute_device)
        attn_weight = cache[f"attn.{layer_idx}.weights"].to(self.compute_device)
        z = cache[f"attn.{layer_idx}.z"].to(self.compute_device)
        ln_scalar = cache[f"ln.attn.{layer_idx}"].to(self.compute_device)
        qn_scalar = cache[f"attn.{layer_idx}.q_norm"].to(self.compute_device) # diff to llama -> qn_scalar
        kn_scalar = cache[f"attn.{layer_idx}.k_norm"].to(self.compute_device) # diff to llama -> kn_scalar

        # Grad Mid Projection
        grad_mid = torch.matmul(grad.to(self.compute_device), o_w).view(batch_size, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2) # (B, H, Q, D)

        # Math: Value Path
        eff_w_v = v_w * norm_w
        v_scalar = (attn_weight * ln_scalar.transpose(-1,-2).unsqueeze(1)).transpose(-2, -1) 
        grad_t = torch.matmul(v_scalar, grad_mid).transpose(1,2)
        v_term = torch.matmul(grad_t.flatten(-2), eff_w_v.flatten(0, 1))

        # 
        term_1 = torch.matmul(grad_mid, v_proj.transpose(-1, -2))
        term_2 = (grad_mid * z).sum(dim=-1, keepdim=True)
        delta_ij = attn_weight * (term_1 - term_2) # Shape (B, H, Q, K)

        # Math Query Pass
        weighted_k_sum = torch.matmul(delta_ij, rot_k_proj) * (self.config.head_dim ** -0.5)
        grad_q_rot = (weighted_k_sum * cos) - (rotate_half(weighted_k_sum) * sin) # + for inverse if sin was - forward
        grad_q_rot = grad_q_rot * qn_scalar # diff to llama -> qn_scalar
        eff_q_w = (q_w * norm_w * qn_w[:, None]).flatten(0, 1) # diff to llama -> qn_w
        q_term = torch.matmul(grad_q_rot.transpose(1, 2).flatten(-2), eff_q_w) * ln_scalar

        weighted_q_sum = torch.matmul(delta_ij.transpose(-2, -1), rot_q_proj) * (self.config.head_dim ** -0.5)
        grad_k_rot = (weighted_q_sum * cos) - (rotate_half(weighted_q_sum) * sin) # + for inverse if sin was - forward
        grad_k_rot = grad_k_rot * kn_scalar # diff to llama -> kn_scalar
        eff_k_w = (k_w * norm_w * kn_w[:, None]).flatten(0, 1) # diff to llama -> kn_w
        k_term = torch.matmul(grad_k_rot.transpose(1, 2).flatten(-2), eff_k_w) * ln_scalar

        return (self.scaling['v'] * v_term + 
                self.scaling['q'] * q_term + 
                self.scaling['k'] * k_term)

    def get_final_norm_scale(self, cache: Dict):
        return cache['final_ln']
    
    def get_component_contributions(self, batch: Dict[str, Any], residual_targets: Dict[str, torch.Tensor]):
        batch_attn = []
        batch_mlp = []
        with torch.no_grad(), self.model.trace(batch):
            for layer_id, layer in enumerate(self.model.model.layers):
                v_proj = layer.self_attn.v_proj.output
                attn_weight = layer.self_attn.output[1]
                o_proj_WT = layer.self_attn.o_proj.weight.data.T
                d_attn = decompose_attention_to_head(
                    attn_weight,
                    v_proj,
                    o_proj_WT,
                    num_attention_heads=self.config.num_attention_heads,
                    num_key_value_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                ) # (B, Q, K, H, D)
                batch_attn.append(
                    torch.einsum('bqd, bq...d -> bq...', residual_targets[f"mid.{layer_id}"].to(self.compute_device), d_attn).save()# (B, Q, K, H)
                )
                del d_attn

                act_prod = layer.mlp.down_proj.input
                down_proj_WT = layer.mlp.down_proj.weight.data.T
                d_mlp = decompose_glu_to_neuron(
                    down_proj_WT,
                    act_prod
                ) # (B, Q, I, D)
                batch_mlp.append(
                    torch.einsum('bqd, bq...d -> bq...', residual_targets[f"mid.{layer_id}"].to(self.compute_device), d_mlp).save() # (B, Q, I)
                )
                del d_mlp
        batch_attn = torch.stack(batch_attn).permute(1, 2, 3, 0, 4).to(self.cache_device) # (B, Q, K, L, H)
        batch_mlp = torch.stack(batch_mlp).permute(1, 2, 0, 3).to(self.cache_device) # (B, Q, L, I)
        return batch_attn, batch_mlp
    
class MistralBackend(Llama2Backend):

    def __init__(self, model, scaling_config = None, cache_device = None):
        super().__init__(model, scaling_config, cache_device)
        model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads
        if model.config.sliding_window != None:
            raise ValueError("Please deactivate sliding window with 'use_sliding_window=False'.")

class Qwen2Backend(Llama2Backend):

    def __init__(self, model, scaling_config = None, cache_device = None):
        super().__init__(model, scaling_config, cache_device)
        model.config.head_dim = model.config.hidden_size // model.config.num_attention_heads

    
BACKEND_MAPPING = {
    'mistralai/Mistral-7B-Instruct-v0.3': MistralBackend,
    'Qwen/Qwen3-4B-Instruct-2507': Qwen3Backend,
    "meta-llama/Llama-2-7b-chat-hf": Llama2Backend,
    'meta-llama/Llama-3.1-8B-Instruct': Llama2Backend,
    'meta-llama/Llama-3.1-8B': Llama2Backend,
    'Qwen/Qwen2.5-32B-Instruct': Qwen2Backend,
}