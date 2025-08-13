from transformers import PreTrainedTokenizerBase
from typing import Union, Optional, overload
import torch
import numpy as np
from collections import defaultdict
import re

class PackedGenericDataItem:

    """Wrapper around a torch tensor that contains multimodal data"""
    def __init__(self, tensor: torch.Tensor, dim_to_pack: int = 0) -> None:
        self.tensor = tensor
        self.dim_to_pack = dim_to_pack
    
    def __call__(self) -> torch.Tensor:
        return self.tensor

    @classmethod
    def concat_packed_items(cls, packed_data: list["PackedGenericDataItem"], return_as_item: bool = False) -> Union["PackedGenericDataBatch", "PackedGenericDataItem"]:
        """Concatenate a list of PackedGenericDataItem objects into a single PackedGenericDataBatch.
        """
        dim = [item.dim_to_pack for item in packed_data]
        assert len(set(dim)) == 1, "All packed multimodal data must have the same dim_to_pack"
        dim = dim[0]
        tensors = [item.tensor for item in packed_data]
        # packed batch
        batch = cls.BatchClass(tensors, dim)
        if return_as_item:
            return batch.as_packed_data_item()
        return batch

class PackedGenericDataBatch:

    ItemClass = PackedGenericDataItem
    """Wrapper around a torch tensor that contains multimodal data"""

    def __init__(self, tensors: Union[list[torch.Tensor], list[PackedGenericDataItem]], dim_to_pack: Optional[int] = None) -> None:
        tensorlist: list[torch.Tensor] = []
        item_dim_to_pack = []
        for item in tensors:
            if isinstance(item, torch.Tensor):
                tensorlist.append(item)
            elif isinstance(item, PackedGenericDataItem):
                item_dim_to_pack.append(item.dim_to_pack)
                tensorlist.append(item.tensor)
            else:
                raise ValueError("tensors must be a list of torch.Tensors or a list of PackedGenericDataItem objects")
        self.tensors: list[torch.Tensor] = tensorlist

        if len(set(item_dim_to_pack)) > 1:
            raise ValueError("All items in the batch must have the same dim_to_pack")
        if len(set(item_dim_to_pack)) == 1:
            assert dim_to_pack is None or item_dim_to_pack[0] == dim_to_pack, "dim_to_pack must not provided, or must be equal as each item in the batch"
            dim_to_pack = item_dim_to_pack[0]
        else:
            # empty string, make sure dim_to_pack is provided
            assert dim_to_pack is not None, "dim_to_pack must be provided for empty batch"
            
        self.dim_to_pack = dim_to_pack 
        
    
    def as_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is not None:
            self.tensors = [item.to(device) for item in self.tensors]
        return torch.cat(self.tensors, dim=self.dim_to_pack).to(device)
    
    def __len__(self) -> int:
        # this is the number of items in the batch
        return len(self.tensors)
    
    def to(self, device: str | torch.device) -> "PackedGenericDataBatch":
        self.tensors = [item.to(device) for item in self.tensors]
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedGenericDataBatch":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        tensors = [self.tensors[i] for i in idx]
        return PackedGenericDataBatch(tensors, self.dim_to_pack)
    
    def as_packed_data_item(self) -> PackedGenericDataItem:
        # you can collapse the batch into a single item by calling this function
        # called inside `message_log_to_flat_messages` to convert the multimodal tensors from turns into a conversation
        ItemClass = self.ItemClass
        return ItemClass(torch.cat(self.tensors, dim=self.dim_to_pack), self.dim_to_pack)
    
    def repeat_interleave(self, num_repeats: int) -> "PackedGenericDataBatch":
        """Repeats the batch num_repeats times."""
        raise NotImplementedError("Why are you interleaving batches? Interleaving is supposed to happen at the message log level. \
                                  If you are adding new functionality, implement this function.")
    
    @classmethod
    def concat_packed_batches(cls, packed_batches: list["PackedGenericDataBatch"]) -> "PackedGenericDataBatch":
        """Concatenate a list of PackedGenericDataBatch objects into a single PackedGenericDataBatch.

        Each batch must have the same dim_to_pack.
        """
        dim_to_packs = [batch.dim_to_pack for batch in packed_batches]
        assert len(set(dim_to_packs)) == 1, "All packed multimodal data must have the same dim_to_pack"
        # concatenate the tensors
        # tensors = [batch.tensor for batch in packed_batches]
        tensors = []
        for batch in packed_batches:
            tensors.extend(batch.tensors)
        dim_to_pack = dim_to_packs[0]
        return cls(tensors, dim_to_pack)

# Set up circular references after both classes are defined
PackedGenericDataItem.BatchClass = PackedGenericDataBatch
PackedGenericDataBatch.ItemClass = PackedGenericDataItem

# define a class for packed multimodal data item
class PackedMultimodalDataItem(PackedGenericDataItem):
    pass

# define a class for packed multimodal data batch
class PackedMultimodalDataBatch(PackedGenericDataBatch):
    pass

# Set up circular references after both classes are defined
PackedMultimodalDataItem.BatchClass = PackedMultimodalDataBatch
PackedMultimodalDataBatch.ItemClass = PackedMultimodalDataItem

def get_multimodal_keys_from_processor(processor) -> list[str]:
    '''
    Get keys of the multimodal data that can be used as model inputs.

    This will be used in the data_processor function to determine which keys to use as model inputs.
    '''
    if isinstance(processor, PreTrainedTokenizerBase):
        return []
    
    all_keys = set()
    if hasattr(processor, "image_processor"):
        all_keys.update(processor.image_processor.model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    # all_keys.update(processor.model_input_names)
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)

        
def get_dim_to_pack_along(processor, key: str) -> int:
    '''
    Special considerations for packing certain keys from certain processors

    In most cases, the packed items are along dim 0
    '''
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0
