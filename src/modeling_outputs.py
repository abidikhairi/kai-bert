from typing import Optional
import torch
from pydantic import BaseModel, ConfigDict


class ModelOutputWithLogits(BaseModel):
    """
    Output class for transformer-based models that include logits and additional hidden state information.

    This class is designed to store model outputs, including the raw logits for predictions, 
    the last hidden states, intermediate hidden states, and attention scores. It leverages 
    `pydantic.BaseModel` for structured validation.

    Attributes:
    -----------
    logits : torch.Tensor
        The output logits from the model, typically used for classification or regression tasks.

    last_hidden_states : Optional[torch.Tensor]
        The final hidden states from the last layer of the transformer model. 
        Useful for feature extraction or downstream tasks.

    hidden_states : Optional[torch.Tensor]
        The hidden states from all layers (if stored), enabling layer-wise analysis of 
        the transformer model.

    attention_scores : Optional[torch.Tensor]
        The attention scores from the self-attention mechanism, which can be used 
        for interpretability and understanding attention distribution.

    Config:
    -------
    model_config = ConfigDict(arbitrary_types_allowed=True)
        Allows arbitrary types like `torch.Tensor` to be stored within the `pydantic.BaseModel`.

    Example Usage:
    --------------
    ```python
    import torch
    from src.modeling_outputs import ModelOutputWithLogits

    output = ModelOutputWithLogits(
        logits=torch.randn(1, 10),
        last_hidden_states=torch.randn(1, 128),
        hidden_states=None,
        attention_scores=None
    )

    print(output.logits.shape)  # Output: torch.Size([1, 10])
    ```
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    logits: torch.Tensor
    last_hidden_states: Optional[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    attention_scores: Optional[torch.Tensor]
