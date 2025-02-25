import math
import torch
from torch import nn
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from kai_bert.configuration import BertConfig
from kai_bert.modeling_outputs import ModelOutputWithLogits


class AbsolutePositionalEncoding(nn.Module):
    """
    Implements absolute positional encoding for transformer models.

    In standard transformer architectures (Vaswani et al., 2017), positional encodings are 
    added to token embeddings to introduce sequential order information, as self-attention 
    layers do not have any inherent notion of token positions.

    This class integrates absolute positional encodings based on sine and cosine functions 
    or learned embeddings, depending on the implementation.

    Parameters:
    -----------
    config : BertConfig
        The configuration object containing model hyperparameters such as `hidden_size` 
        and `max_position_embeddings`.

    References:
    -----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
      "Attention is All You Need". NeurIPS.
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.

    Example Usage:
    --------------
    ```python
    from torch import nn
    from src.modeling import AbsolutePositionalEncoding, BertConfig

    config = BertConfig(hidden_size=128, max_position_embeddings=1026)
    pos_encoding = AbsolutePositionalEncoding(config)
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        
        pe = torch.zeros(self.max_position_embeddings, self.hidden_size)
        
        position = torch.arange(0, self.max_position_embeddings, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.pow(10000, -torch.arange(0, self.hidden_size, 2).float() / self.hidden_size)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)        
        
    def forward(
        self,
        hidden_states: torch.Tensor  
    ) -> torch.Tensor:
        seq_len = hidden_states.size(1)

        return hidden_states + self.pe[:, :seq_len, :]

    def __repr__(self):
        return f"PositionalEncoding(hidden_size={self.hidden_size}, max_position_embeds={self.max_position_embeddings})"


class BertEmbedding(nn.Module):
    """
    Embedding layer for a BERT-like transformer model.

    This class implements the token embeddings, positional embeddings, and optionally 
    segment (token type) embeddings used in the BERT architecture. The embeddings 
    are summed and passed through a normalization layer before being fed into 
    the transformer encoder.

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as `vocab_size`, 
        `hidden_size`, and `max_position_embeddings`.

    Attributes:
    -----------
    word_embeddings : nn.Embedding
        Token embedding layer that maps input token indices to dense vectors.
    
    position_embeddings : nn.Embedding
        Positional encoding layer that assigns each token a position-dependent vector.

    token_type_embeddings : nn.Embedding (optional)
        Segment embeddings that distinguish between different sequences 
        (e.g., in sentence-pair tasks like QA).

    layer_norm : nn.LayerNorm
        Normalization layer applied after summing embeddings.

    dropout : nn.Dropout
        Dropout layer applied for regularization.

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
    import torch
    from src.modeling import BertEmbedding, BertConfig

    config = BertConfig(vocab_size=30000, hidden_size=128, max_position_embeddings=1026)
    embedding_layer = BertEmbedding(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 128))  # Simulated input sequence
    embeddings = embedding_layer(input_ids)
    print(embeddings.shape)  # Expected output: torch.Size([1, 128, config.hidden_size])
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.token_embeds = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        self.position_embeds = AbsolutePositionalEncoding(config)
        
    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:

        # return self.position_embeds(self.token_embeds(input_ids))
        return self.token_embeds(input_ids)


class SelfAttention(nn.Module):
    """
    Implements the self-attention mechanism used in transformer architectures.

    Self-attention allows the model to weigh input tokens differently based on their 
    relevance to each other, enabling contextual understanding across the sequence.

    This class typically includes:
    - Query, Key, and Value projections
    - Scaled dot-product attention computation
    - Optional dropout for regularization

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as `hidden_size` 
        and `num_attention_heads`.

    Attributes:
    -----------
    query : nn.Linear
        Linear layer that projects input embeddings to query representations.
    
    key : nn.Linear
        Linear layer that projects input embeddings to key representations.

    value : nn.Linear
        Linear layer that projects input embeddings to value representations.

    attention_dropout : nn.Dropout
        Dropout applied to the attention weights for regularization.

    softmax : nn.Softmax(dim=-1)
        Softmax function to compute attention scores.

    References:
    -----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
      "Attention is All You Need". NeurIPS.
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.

    Example Usage:
    --------------
    ```python
    import torch
    from src.modeling import SelfAttention, BertConfig

    config = BertConfig(hidden_size=128, num_attention_heads=4)
    attention_layer = SelfAttention(config)

    input_tensor = torch.randn(1, 128, config.hidden_size)  # Batch of 1, sequence length 128
    output = attention_layer(input_tensor)
    print(output.shape)  # Expected output: torch.Size([1, 128, config.hidden_size])
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        
        self.query = nn.Linear(self.attention_head_size, self.attention_head_size, bias=False)        
        self.key = nn.Linear(self.attention_head_size, self.attention_head_size, bias=False)        
        self.value = nn.Linear(self.attention_head_size, self.attention_head_size, bias=False)        
        
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)        
    
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshapes and transposes the input tensor for multi-head attention computations.

        This function reshapes the input tensor from `[batch_size, seq_len, hidden_size]` 
        to `[batch_size, num_heads, seq_len, head_dim]`, where `head_dim` is derived from 
        `hidden_size // num_attention_heads`. The final transpose ensures that attention 
        operations are performed efficiently across different heads.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor of shape `[batch_size, seq_len, hidden_size]`, 
            typically the output of a linear projection (query, key, or value).

        Returns:
        --------
        torch.Tensor
            The reshaped and transposed tensor of shape `[batch_size, num_heads, seq_len, head_dim]`, 
            ready for scaled dot-product attention computations.

        Example Usage:
        --------------
        ```python
        import torch
        from src.modeling import SelfAttention

        attention_layer = SelfAttention(config)
        
        x = torch.randn(1, 128, config.hidden_size)  # Batch size = 1, sequence length = 128
        transformed_x = attention_layer._transpose_for_scores(x)

        print(transformed_x.shape)  # Expected output: torch.Size([1, num_heads, 128, head_dim])
        ```
        """
        bs, seq_len, _ = x.shape
        new_x_shape = (bs, seq_len) + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)

        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        bs, seq_len, _ = hidden_states.shape
        
        query_states = self._transpose_for_scores(hidden_states)
        key_states = self._transpose_for_scores(hidden_states)
        value_states = self._transpose_for_scores(hidden_states)
        
        query_states = self.query(query_states)
        key_states = self.key(key_states)
        value_states = self.value(value_states)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        context = torch.matmul(attention_scores, value_states)
        context = context.permute(0, 2, 1, 3).contiguous()
        
        context = context.view(bs, seq_len, -1)
        context = self.output(context)
        
        result = (context, attention_scores)
        
        return result

class BertMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used in the feed-forward network (FFN) of a transformer block.

    This class implements the two-layer feed-forward network (FFN) that follows the 
    self-attention mechanism in transformer models. The MLP consists of:
    - A linear transformation to expand the hidden dimension.
    - A non-linear activation function.
    - A second linear transformation to project back to the original hidden size.
    - Optional dropout for regularization.

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as:
        - `hidden_size`: Dimensionality of the model embeddings.
        - `intermediate_size`: Size of the hidden layer in the MLP.
        - `hidden_act`: Activation function to use (e.g., ReLU, GELU).
        - `hidden_dropout_prob`: Dropout probability for regularization.

    Attributes:
    -----------
    fc1 : nn.Linear
        First linear layer that expands the hidden representation from `hidden_size` to `intermediate_size`.

    activation : nn.Module
        Non-linear activation function (e.g., ReLU, GELU).

    fc2 : nn.Linear
        Second linear layer that projects the intermediate representation back to `hidden_size`.

    dropout : nn.Dropout
        Dropout layer applied after the second linear transformation.

    References:
    -----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
      "Attention is All You Need". NeurIPS.
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.

    Example Usage:
    --------------
    ```python
    import torch
    from src.modeling import BertMLP, BertConfig

    config = BertConfig(hidden_size=128, intermediate_size=512, hidden_act='relu')
    mlp_layer = BertMLP(config)

    input_tensor = torch.randn(1, 128, config.hidden_size)  # Batch of 1, sequence length 128
    output = mlp_layer(input_tensor)

    print(output.shape)  # Expected output: torch.Size([1, 128, config.hidden_size])
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        
        x = self.linear1(hidden_states)
        x = self.act(x)
        x = self.linear2(x)
        
        return x


class BertLayer(nn.Module):
    """
    A single Transformer encoder layer used in BERT.

    This layer consists of:
    - A multi-head self-attention mechanism.
    - A feed-forward network (MLP).
    - Layer normalization and dropout for regularization.
    - Residual connections for stability.

    Each `BertLayer` processes an input sequence and outputs a transformed representation 
    that captures contextual dependencies across the sequence.

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as:
        - `hidden_size`: Dimensionality of the model embeddings.
        - `num_attention_heads`: Number of attention heads in self-attention.
        - `intermediate_size`: Size of the feed-forward hidden layer.
        - `hidden_dropout_prob`: Dropout probability applied to activations.
        - `attention_probs_dropout_prob`: Dropout probability applied to attention scores.

    Attributes:
    -----------
    attention : SelfAttention
        Multi-head self-attention module.

    attention_output_norm : nn.LayerNorm
        Layer normalization applied after the attention mechanism.

    mlp : BertMLP
        Feed-forward network applied after attention.

    output_norm : nn.LayerNorm
        Layer normalization applied after the feed-forward network.

    dropout : nn.Dropout
        Dropout applied at various points for regularization.

    References:
    -----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
      "Attention is All You Need". NeurIPS.
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.

    Example Usage:
    --------------
    ```python
    import torch
    from src.modeling import BertLayer, BertConfig

    config = BertConfig(hidden_size=128, num_attention_heads=4, intermediate_size=512)
    bert_layer = BertLayer(config)

    input_tensor = torch.randn(1, 128, config.hidden_size)  # Batch of 1, sequence length 128
    attention_mask = torch.ones(1, 1, 1, 128)  # Example mask for attention

    output = bert_layer(input_tensor, attention_mask)
    print(output.shape)  # Expected output: torch.Size([1, 128, config.hidden_size])
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.pre_attn_layer_norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.self_attn = SelfAttention(config)
        self.post_attn_layer_norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.mlp = BertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ):
        residual = self.pre_attn_layer_norm(hidden_states)
        residual, attentions = self.self_attn(residual, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = self.post_attn_layer_norm(hidden_states)
        residual = self.mlp(residual)        
        hidden_states = residual + hidden_states
        
        result = (hidden_states, attentions) if output_attentions else (hidden_states,)
        
        return result

class BertLMHead(nn.Module):
    """
    Language Model (LM) head for BERT-based models.

    This class implements the output head used to predict the next token in a sequence,
    or to perform masked language modeling (MLM). It consists of:
    - A linear transformation from the hidden representation to the vocabulary size.
    - Softmax activation applied to the logits to obtain token probabilities.
    - Optionally, a masking mechanism for predicting masked tokens in a given input.

    The LM head is typically used for pretraining BERT models using masked language modeling,
    where a portion of the input tokens are randomly masked, and the model learns to predict 
    these tokens based on context.

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as:
        - `hidden_size`: Dimensionality of the model's hidden states.
        - `vocab_size`: Size of the vocabulary.
        - `hidden_act`: Activation function used in the hidden layers.
        - `layer_norm_eps`: Epsilon for layer normalization (if applicable).

    Attributes:
    -----------
    dense : nn.Linear
        Linear transformation from the hidden representation to logits with shape `[batch_size, seq_len, vocab_size]`.

    layer_norm : nn.LayerNorm
        Layer normalization applied to the logits.

    decoder : nn.Linear
        Decoder layer used to map the hidden representation to vocabulary size.

    bias : Optional[torch.Tensor]
        Optional bias term for the linear transformation.

    References:
    -----------
    - Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). 
      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". 
      arXiv preprint arXiv:1810.04805.
    - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). 
      "Improving Language Understanding by Generative Pre-Training". 
      OpenAI.

    Example Usage:
    --------------
    ```python
    import torch
    from src.modeling import BertLMHead, BertConfig

    config = BertConfig(hidden_size=128, vocab_size=25000)
    lm_head = BertLMHead(config)

    input_tensor = torch.randn(1, 128, config.hidden_size)  # Batch of 1, sequence length 128
    logits = lm_head(input_tensor)

    print(logits.shape)  # Expected output: torch.Size([1, 128, config.vocab_size])
    ```
    """
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(hidden_states)


class BertModel(PreTrainedModel):
    """
    Core BERT (Bidirectional Encoder Representations from Transformers) model.

    This class implements the transformer-based encoder architecture introduced in BERT.
    It consists of:
    - Token embeddings, positional embeddings, and segment embeddings.
    - A stack of transformer encoder layers (`BertLayer`).
    - An optional output layer for downstream tasks.

    This model serves as a backbone for various NLP applications, including text classification, 
    named entity recognition, and sequence-to-sequence learning.

    Parameters:
    -----------
    config : BertConfig
        Configuration object containing model hyperparameters such as:
        - `vocab_size`: Size of the vocabulary.
        - `hidden_size`: Embedding dimensionality.
        - `num_hidden_layers`: Number of transformer layers.
        - `num_attention_heads`: Number of attention heads.
        - `intermediate_size`: Size of the feed-forward network.
        - `max_position_embeddings`: Maximum sequence length supported.
        - `hidden_dropout_prob`: Dropout probability for activations.
        - `attention_probs_dropout_prob`: Dropout probability for attention scores.

    Attributes:
    -----------
    embeddings : BertEmbedding
        Embedding layer that combines token, positional, and segment embeddings.

    encoder : nn.ModuleList[BertLayer]
        A stack of transformer encoder layers that process input sequences.

    pooler : nn.Linear (Optional)
        An optional layer that extracts the representation of the `[CLS]` token 
        for sentence-level tasks.

    dropout : nn.Dropout
        Dropout applied to prevent overfitting.

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
    import torch
    from src.modeling import BertModel, BertConfig

    config = BertConfig(vocab_size=25000, hidden_size=128, num_hidden_layers=6)
    model = BertModel(config)

    input_ids = torch.randint(0, config.vocab_size, (1, 128))  # Batch of 1, sequence length 128
    attention_mask = torch.ones(1, 128)  # Masking valid tokens

    output = model(input_ids, attention_mask=attention_mask)
    print(output.last_hidden_state.shape)  # Expected output: torch.Size([1, 128, config.hidden_size])
    ```
    """    
    config_class = BertConfig
    
    def __init__(self, config: BertConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.embed_tokens = BertEmbedding(config)
        
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.lm_head = BertLMHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = False
    ) -> ModelOutputWithLogits:
        hidden_states = self.embed_tokens(input_ids)
        
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions is True:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,) 

        last_hidden_states = hidden_states
        logits = self.lm_head(last_hidden_states)

        return ModelOutputWithLogits(
            logits=logits,
            last_hidden_states=last_hidden_states,
            hidden_states=all_hidden_states,
            attention_scores=all_attentions
        )
 