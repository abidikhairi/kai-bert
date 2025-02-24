from transformers import PretrainedConfig


class BertConfig(PretrainedConfig):
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
