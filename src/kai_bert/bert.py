import torch
from torch import (
    nn,
    optim
)
from typing import List, Optional, Tuple, Union
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, PreTrainedTokenizer

from kai_bert.configuration import BertConfig
from kai_bert.modeling import BertModel


class BertModelForMaskedLM(LightningModule):
    """
    BERT-based Masked Language Model for pretraining with PyTorch Lightning.

    This class implements the BERT architecture for masked language modeling (MLM) using 
    PyTorch Lightning. It can be used for pretraining or fine-tuning a BERT model on a 
    masked language modeling task where a portion of input tokens are randomly masked and 
    the model learns to predict the masked tokens based on their context.

    Parameters:
    -----------
    config : Optional[BertConfig]
        Configuration object containing model hyperparameters such as:
        - `vocab_size`: Size of the vocabulary.
        - `hidden_size`: Dimensionality of the model's hidden states.
        - `num_attention_heads`: Number of attention heads.
        - `num_hidden_layers`: Number of transformer layers.
        - `max_position_embeddings`: Maximum sequence length.
        - `hidden_dropout_prob`: Dropout probability for hidden layers.
        - `attention_probs_dropout_prob`: Dropout probability for attention probabilities.

    model_path_or_id : Optional[str]
        Path or identifier to load a pretrained BERT model.

    tokenizer : Union[str, PreTrainedTokenizer]
        Tokenizer to convert text to token ids and vice versa. Can be a pretrained tokenizer or a path to one.

    mask_ratio : float, default=0.15
        The ratio of tokens in the input sequence to be masked for the MLM task.

    learning_rate : float, default=1e-3
        Learning rate used by the AdamW optimizer.

    beta1 : float, default=0.99
        Beta1 parameter for the AdamW optimizer.

    beta2 : float, default=0.98
        Beta2 parameter for the AdamW optimizer.

    epsilon : float, default=1e-6
        Epsilon parameter for the AdamW optimizer.

    Attributes:
    -----------
    bert : BertModel
        The BERT model backbone, either loaded from a pretrained model or initialized from scratch.

    tokenizer : PreTrainedTokenizer
        Tokenizer used to process input text into token ids.

    loss_fn : nn.CrossEntropyLoss
        Loss function used for training the masked language model.

    mask_ratio : float
        Masking ratio used to select tokens for masking during training.

    learning_rate : float
        The learning rate for optimizer configuration.

    beta1 : float
        Beta1 parameter for the AdamW optimizer.

    beta2 : float
        Beta2 parameter for the AdamW optimizer.

    epsilon : float
        Epsilon parameter for the AdamW optimizer.

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
    from src.bert import BertModelForMaskedLM, BertConfig

    config = BertConfig(vocab_size=25000, hidden_size=128, num_hidden_layers=6)
    model = BertModelForMaskedLM(config=config)

    # Example input for training
    input_ids = torch.randint(0, config.vocab_size, (2, 128))  # Batch of 2, sequence length 128
    attention_mask = torch.ones(2, 128)  # Masking valid tokens
    labels = torch.randint(0, config.vocab_size, (2, 128))

    # Forward pass
    loss = model.training_step({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, batch_idx=0)
    print(f"Training Loss: {loss}")
    ```
    """
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
        """
        Mask a portion of the input tokens (used for MLM) by replacing them with 
        the `[MASK]` token and generating the corresponding labels.
        
        Args:
            input_ids: Tensor of token ids for the input sequence.
        
        Returns:
            masked_input_ids: Input tensor with masked tokens.
            labels: Labels for the masked tokens.
        """
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
