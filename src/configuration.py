from transformers import PretrainedConfig


class BertConfig(PretrainedConfig):
    """
    Configuration class for a BERT-like transformer model.

    This class defines the model architecture hyperparameters, including the vocabulary size, 
    hidden dimensions, number of attention heads, and dropout rates. It is a subclass of 
    `PretrainedConfig` and is used to instantiate a model with specific configurations.

    Parameters:
    -----------
    vocab_size : int, default=25000
        The size of the vocabulary, defining the number of unique tokens in the tokenizer.
    
    hidden_size : int, default=128
        The dimension of the hidden layers in the transformer model.
    
    num_hidden_layers : int, default=6
        The number of transformer encoder layers.
    
    num_attention_heads : int, default=4
        The number of attention heads in each self-attention layer.
    
    intermediate_size : int, default=512
        The size of the feed-forward network within each transformer layer.
    
    hidden_act : str, default='relu'
        The activation function used in the feed-forward layers. Common choices include 'relu' and 'gelu'.
    
    hidden_dropout_prob : float, default=0.1
        The dropout probability applied to hidden states for regularization.
    
    attention_probs_dropout_prob : float, default=0.1
        The dropout probability applied to attention scores.
    
    max_position_embeddings : int, default=1026
        The maximum sequence length that the model can handle.
    
    pad_token_id : int, default=1
        The token ID used for padding sequences.
    
    mask_token_id : int, default=2
        The token ID used for masked tokens (e.g., in masked language modeling).

    References:
    -----------
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
      "Attention is All You Need". NeurIPS.

    Example Usage:
    --------------
    ```python
    from transformers import BertConfig
    config = BertConfig(vocab_size=30000, hidden_size=256, num_hidden_layers=8)
    ```
    """
    def __init__(
        self,
        vocab_size: int = 25000,
        hidden_size: int = 128,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 4,
        intermediate_size: int = 512,
        hidden_act: str = 'relu',
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1026,
        pad_token_id: int = 1,
        mask_token_id: int = 2,
        *args,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, *args, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
