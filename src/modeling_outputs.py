from typing import Optional
import torch
from pydantic import BaseModel, ConfigDict


class ModelOutputWithLogits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    logits: torch.Tensor
    last_hidden_states: Optional[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    attention_scores: Optional[torch.Tensor]
