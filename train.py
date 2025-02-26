import argparse

from pytorch_lightning import (
    Trainer,
    loggers,
    callbacks                               
)

from kai_bert import LanguageModelingDataModule
from kai_bert import BertModelForMaskedLM
from kai_bert.configuration import BertConfig

def train(args):
    args_dict = vars(args)
    
    model_config_path = args_dict["model_config_path"]
    model_path_or_id = args_dict["model_path_or_id"]
    tokenizer = args_dict["tokenizer"]
    mask_ratio = args_dict["mask_ratio"]
    learning_rate = args_dict["learning_rate"]
    beta1 = args_dict["beta1"]
    beta2 = args_dict["beta2"]
    epsilon = args_dict["epsilon"]
    train_file = args_dict["train_file"]
    valid_file = args_dict["valid_file"]
    test_file = args_dict["test_file"]
    text_column_name = args_dict["text_column_name"]
    batch_size = args_dict["batch_size"]
    num_proc = args_dict["num_proc"]
    accelerator = args_dict['accelerator']
    max_steps = args_dict['max_steps']
    experim_name = args_dict['experim_name']
    
    data_module = LanguageModelingDataModule(
        tokenizer=tokenizer,
        train_file_path=train_file,
        valid_file_path=valid_file,
        test_file_path=test_file,
        text_column_name=text_column_name,
        batch_size=batch_size,
        num_proc=num_proc
    )
    
    if model_config_path is not None:
        config = BertConfig.from_json_file(model_config_path)
    else:
        config = None
    
    model = BertModelForMaskedLM(
        config=config,
        model_path_or_id=model_path_or_id,
        tokenizer=tokenizer,
        mask_ratio=mask_ratio,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    
    log_dir = f'data/{experim_name}/logs'
    ckpt_dir = f'data/{experim_name}/ckpts'
    
    csv_logger = loggers.CSVLogger(save_dir=log_dir, name=experim_name)
    ckpt_callback = callbacks.ModelCheckpoint(dirpath=ckpt_dir, monitor='val_loss', save_top_k=True)
    
    trainer = Trainer(
        accelerator=accelerator,
        max_steps=max_steps,
        callbacks=[ckpt_callback],
        logger=[csv_logger],
        log_every_n_steps=500
    )
    
    trainer.fit(model, data_module)
    
    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kai-Bert: An educational BERT implementation")
    
    parser.add_argument('--accelerator', type=str, default='cpu', required=False, help="Device to use for training. Options: 'cpu', 'gpu', 'tpu'. Defaults to 'cpu'.")
    parser.add_argument('--max-steps', type=int, default=20000, required=False, help="Maximum number of training steps before stopping. Defaults to 20000.")
    parser.add_argument('--experim-name', type=str, default='kai-bert', required=False, help="Experiment name for logging and tracking results. Defaults to 'kai-bert'.")

    parser.add_argument('--model-config-path', type=str, required=None, help="Path to model_config.json")
    parser.add_argument("--model-path-or-id", type=str, default=None, help="Path or ID of the pretrained model")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path or model name")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Masking ratio for MLM. Defaults to 0.15")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training. Defaults to 1e-3")
    parser.add_argument("--beta1", type=float, default=0.99, help="Beta1 parameter for Adam optimizer. Defaults to 0.99")
    parser.add_argument("--beta2", type=float, default=0.98, help="Beta2 parameter for Adam optimizer. Defaults to 0.98")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Epsilon value for numerical stability in Adam. Defaults to 1e-6")

    parser.add_argument("--train-file", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--valid-file", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--test-file", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--text-column-name", type=str, default='text', required=True, help="Name of the text column in the dataset. Defaults `text`")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training. Defaults 32")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for data loading. Defaults 1")

    args = parser.parse_args()
    
    if not (args.model_config_path or args.model_path_or_id):
        parser.error("At least one of --model-config-path or --model-path-or-id must be provided.")

    train(args)