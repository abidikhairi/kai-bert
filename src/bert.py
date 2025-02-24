import torch
from torch import (
    nn,
    optim
)
from typing import List, Optional, Tuple, Union
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.configuration import BertConfig
from src.modeling import BertModel


class BertModelForMaskedLM(LightningModule):
    def __init__(
        self,
        config: Optional[BertConfig] = None,
        model_path_or_id: Optional[str] = None,
        tokenizer: Union[str, PreTrainedTokenizer] = None,
        mask_ratio: float = 0.15,
        learning_rate: float = 1e-3,
        beta1: float = 0.99,
        beta2: float = 0.98,
        epsilon: float = 1e-6,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        if config is None and model_path_or_id is None:
            raise ValueError("Either `config` or `model_path_or_id` must be provided.")

        if config is None:
            config = BertConfig.from_pretrained(model_path_or_id)

        self.config = config
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        if model_path_or_id:
            self.bert = BertModel.from_pretrained(model_path_or_id)
        else:
            self.bert = BertModel(config)

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.loss_fn = nn.CrossEntropyLoss()
    
        self.save_hyperparameters()
    
    def forward(self, *args, **kwargs):
        return self.bert(*args, **kwargs)
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.bert.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            eps=self.epsilon
        )

        return {
            "optimizer": optimizer
        }
    
    def _mask_inputs(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prob_matrix = torch.full(input_ids.shape, self.mask_ratio, device=input_ids.device)
        mask_indices = torch.bernoulli(prob_matrix).bool()
        
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()

        rand_tensor = torch.rand(input_ids.shape, device=input_ids.device)

        masked_input_ids[mask_indices & (rand_tensor < 0.8)] = self.tokenizer.mask_token_id

        random_tokens = torch.randint_like(input_ids, low=0, high=self.tokenizer.vocab_size)
        masked_input_ids[mask_indices & (rand_tensor >= 0.8) & (rand_tensor < 0.9)] = random_tokens[mask_indices & (rand_tensor >= 0.8) & (rand_tensor < 0.9)]
        
        labels[~mask_indices] = -100
        
        return masked_input_ids, labels
    
    def _build_4d_attention_mask(
        self,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, :, None] * attention_mask[:, None, :]
        
        extended_attention_mask = extended_attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.expand(-1, self.config.num_attention_heads, -1, -1)
        extended_attention_mask = (1 - extended_attention_mask) * 1e-8
    
        extended_attention_mask = extended_attention_mask.to(attention_mask.device)
        
        return extended_attention_mask
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        input_ids, labels = self._mask_inputs(input_ids)
        attention_mask = self._build_4d_attention_mask(attention_mask)
        
        outputs = self.forward(input_ids, attention_mask)
        
        logits = outputs.logits
        
        loss = self.loss_fn(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask= batch['attention_mask']
        
        input_ids, labels = self._mask_inputs(input_ids)
        attention_mask = self._build_4d_attention_mask(attention_mask)
        
        outputs = self.forward(input_ids, attention_mask)
        
        logits = outputs.logits
        
        loss = self.loss_fn(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        
        self.log("valid_loss", loss, prog_bar=True, logger=True)
        self.log("valid_ppl", loss.exp(), prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask= batch['attention_mask']
        labels = batch['labels']
        
        attention_mask = self._build_4d_attention_mask(attention_mask)
        
        outputs = self.forward(input_ids, attention_mask)
        
        logits = outputs.logits
        
        loss = self.loss_fn(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_ppl", loss.exp(), prog_bar=True, logger=True)
        
        return loss
    
    @torch.inference_mode()
    def predict_step(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, padding_side='left')
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        attention_mask = self._build_4d_attention_mask(attention_mask)
        
        