from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from tracer.backend import ModelBackend


class BaseTracer(ABC):
    """
    Abstract base class for tracing model behaviors and extracting attribution scores.
    """
    
    def __init__(self, backend: ModelBackend, tokenizer: PreTrainedTokenizer) -> None:
        """
        Initializes the tracer.

        Args:
            backend (ModelBackend): The backend wrapper handling model operations.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        """
        self.backend = backend
        self.tokenizer = tokenizer
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def trace(self, prompts: List[str], targets: Union[List[str], List[int], torch.Tensor], batch_size: int = 16) -> List[Any]:
        """
        Executes the tracing process over a sequence of prompts and targets.

        Args:
            prompts (List[str]): The list of input text prompts.
            targets (Union[List[str], List[int], torch.Tensor]): The corresponding targets for attribution.
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 16.

        Returns:
            List[Any]: A list containing the extracted contribution scores for each item.
        """
        if isinstance(targets, list):
            targets = self._resolve_targets(targets)
            
        dl = self._create_dataloader(prompts, targets, batch_size)
        contributions = []

        for batch in tqdm(dl):
            batch_contributions = self.batch_trace(batch)

            contribution_scores = self._extract_from_mask(batch_contributions, batch['attention_mask'])
            contributions.extend(contribution_scores)

        return contributions
    
    @abstractmethod
    def batch_trace(self, batch: Dict[str, Any]) -> Any:
        """
        Processes a single batch to compute gradients/contributions.

        Args:
            batch (Dict[str, Any]): A dictionary containing batch inputs and targets.

        Returns:
            Any: The raw batch contributions prior to mask extraction.
        """
        pass

    @abstractmethod
    def _extract_from_mask(self, tensors: Any, mask: torch.Tensor) -> List[Any]:
        """Helper to extract context-only tokens from flat tensors based on masks."""
        pass

    def _init_gradients(self, batch: Dict[str, Any], cache: Dict[str, Any]) -> torch.Tensor:
        """
        Initializes the gradient vector at the output of the model.

        Args:
            batch (Dict[str, Any]): The current batch dictionary.
            cache (Dict[str, Any]): The forward pass activation cache.

        Returns:
            torch.Tensor: The initialized gradient tensor.
        """
        B, S = batch['input_ids'].shape
        target_ids = batch['targets']
        
        # Start with empty vectors
        grad = torch.zeros(B, S, self.backend.config.hidden_size, 
                           device=self.backend.compute_device, dtype=self.backend.dtype)
        
        # Get LM Head Weights
        device = self.backend.model.lm_head.device
        lm_head_W = self.backend.model.lm_head.weight.data[target_ids.to(device)]
        
        # Apply final norm scaling
        final_scale = self.backend.get_final_norm_scale(cache).to(device)
        norm_W = self.backend.model.model.norm.weight.data.to(device)
        
        grad[:, -1] = lm_head_W * norm_W * final_scale
        return grad
    
    def _resolve_targets(self, targets: List[Union[str, int]]) -> torch.Tensor:
        """
        Converts a list of strings or integers into a tensor of target Token IDs.

        Args:
            targets (List[Union[str, int]]): A list of target strings or token IDs.

        Returns:
            torch.Tensor: A 1D tensor of target token IDs.
        """
        # If targets are strings, tokenize them
        if isinstance(targets[0], str):
            target_ids = [
                self.tokenizer.encode(t, add_special_tokens=False)[0] 
                for t in targets
            ]
        else:
            target_ids = targets

        return torch.tensor(target_ids, dtype=torch.long)

    def _create_dataloader(
        self, 
        prompts: List[str], 
        targets: torch.Tensor, 
        batch_size: int
    ) -> DataLoader:
        """
        Creates a DataLoader that handles tokenization padding and batching.

        Args:
            prompts (List[str]): Input prompts.
            targets (torch.Tensor): Encoded target IDs or target tensors.
            batch_size (int): Size of the batches.

        Returns:
            DataLoader: A PyTorch DataLoader yielding prepared batches.
        """
        
        def collate_fn(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch_prompts = [item['prompt'] for item in batch_items]
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors='pt', 
                padding=True
            )
            batch_targets = torch.stack([item['target'] for item in batch_items])

            inputs["targets"] = batch_targets
            return dict(inputs)

        dataset = [
            {'prompt': p, 'target': t} 
            for p, t in zip(prompts, targets)
        ]

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn, 
            shuffle=False
        )
    
class InputTracer(BaseTracer):
    """
    Tracer implementation that attributes outputs directly to input embeddings.
    """

    def batch_trace(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor]:
        """
        Computes the input embedding contributions for a batch.

        Args:
            batch (Dict[str, Any]): The input batch dictionary.

        Returns:
            Tuple[torch.Tensor]: A tuple containing the flattened contribution tensor.
        """
        cache = self.backend.run_forward_and_cache(batch)
        
        if batch['targets'].dtype == torch.long:
            batch_grad = self._init_gradients(batch, cache)
        else:
            batch_grad = batch['targets'].to(self.backend.compute_device)
        
        for i in range(self.backend.config.num_hidden_layers - 1, -1, -1):
            mlp_update = self.backend.get_mlp_update(i, batch_grad, cache)
            batch_grad += mlp_update
            
            attn_update = self.backend.get_attn_update(i, batch_grad, cache)
            batch_grad += attn_update
        
        emb = cache['emb']
        if emb.device != batch_grad.device:
            emb = emb.to(batch_grad.device)
        batch_contributions = torch.matmul(emb.unsqueeze(2), batch_grad.unsqueeze(3)).flatten(1).cpu()
        
        return (batch_contributions,)

    def _extract_from_mask(self, tensors: Tuple[torch.Tensor], mask: torch.Tensor) -> List[List[float]]:
        """
        Helper to extract context-only tokens from flat tensors based on masks.

        Args:
            tensors (Tuple[torch.Tensor]): The batch contribution tensors.
            mask (torch.Tensor): The attention mask indicating valid tokens.

        Returns:
            List[List[float]]: A nested list of valid token contributions per sequence.
        """
        lengths = mask.sum(dim=1).tolist()
        flat_tensor = tensors[0][mask.bool().to(tensors[0].device)]
        return [t.tolist() for t in torch.split(flat_tensor, lengths)]
    
class ComponentTracer(BaseTracer):
    """
    Tracer implementation that attributes outputs to specific Attention and MLP components 
    across all transformer layers.
    """

    def batch_trace(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the layer-wise component contributions for a batch.

        Args:
            batch (Dict[str, Any]): The input batch dictionary.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing attention and MLP contribution tensors.
        """
        cache = self.backend.run_forward_and_cache(batch)
        
        if batch['targets'].dtype == torch.long:
            batch_grad = self._init_gradients(batch, cache)
        else:
            batch_grad = batch['targets'].to(self.backend.compute_device)
        
        residual_targets = {}
        for i in range(self.backend.config.num_hidden_layers - 1, -1, -1):
            residual_targets[f"post.{i}"] = batch_grad.clone().cpu()
            mlp_update = self.backend.get_mlp_update(i, batch_grad, cache)
            batch_grad += mlp_update
            
            residual_targets[f"mid.{i}"] = batch_grad.clone().cpu()
            attn_update = self.backend.get_attn_update(i, batch_grad, cache)
            batch_grad += attn_update
        
        batch_attn, batch_mlp = self.backend.get_component_contributions(batch, residual_targets) # (B, Q, K, L, H), (B, Q, L, I)
        
        return batch_attn, batch_mlp

    
    def _extract_from_mask(self, tensors: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> List[Tuple[List[Any], List[Any]]]:
        """
        Helper to extract context-only tokens from flat tensors based on masks.

        Args:
            tensors (Tuple[torch.Tensor, torch.Tensor]): The attention and MLP contribution tensors.
            mask (torch.Tensor): The attention mask indicating valid tokens.

        Returns:
            List[Tuple[List[Any], List[Any]]]: A list mapping valid attention and MLP components.
        """
        attn_tensor, mlp_tensor = tensors
        length = mask.sum(dim=1)

        # extract attn contributions
        num_layer, num_heads = attn_tensor.shape[-2:]
        attn_mask = (mask[:, None, :] * mask[:, :, None]).bool()
        flat_attn_tensor= attn_tensor[attn_mask.to(attn_tensor.device)]
        attn_lists = [
            t.view(l, l, num_layer, num_heads).tolist() for t, l in zip(
                torch.split(flat_attn_tensor, (length ** 2).tolist()), length.tolist()
            )
        ]

        # extract mlp contributions
        flat_mlp_tensor= mlp_tensor[mask.bool().to(mlp_tensor.device)]
        mlp_lists = [t.tolist() for t in torch.split(flat_mlp_tensor, length.tolist())]

        return list(zip(attn_lists, mlp_lists))