import math
import torch
from torch import nn
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from src.configuration import BertConfig
from src.modeling_outputs import ModelOutputWithLogits


class AbsolutePositionalEncoding(nn.Module):
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
    def __init__(self, config: BertConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(hidden_states)


class BertModel(PreTrainedModel):
    
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
 