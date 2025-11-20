import torchtext
torchtext.disable_torchtext_deprecation_warning()
from abc import ABC, abstractmethod

from pathlib import Path
import pandas as pd

import torch as th
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torch.cuda.amp import autocast

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import InputMode
from mamkit.data.processing import UnimodalProcessor
from mamkit.data.collators import UnimodalCollator, TextTransformerCollator
from mamkit.utility.model import MAMKitLightingModel
from mamkit.utility.callbacks import PycharmProgressBar

try:
    from src import io_utils as io
except ModuleNotFoundError:
    import io_utils as io

import logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
     
    def __init__(self, config_cls, model_cls, model_name, task_name, dataset, metric_dict, trainer_args, num_classes, tags):
        self.task_name = task_name

        self.model_cls = model_cls
        self.model_name = model_name
        self.metric_dict = metric_dict
        self.trainer_args = trainer_args
        
        self.config = config_cls.from_config(
            key=ConfigKey(
                dataset=dataset,
                task_name=self.task_name,
                input_mode=InputMode.TEXT_ONLY,
                tags=tags,
            )
        )
        self.config.num_classes = num_classes

        self.model = None
        self.processor = None
        self.loader = None

        project_root = Path(__file__).resolve().parents[2]
        self.mamkit_dir = project_root / 'mamkit_data'

        output_dir = project_root / 'artifacts'
        self.model_dir = output_dir / 'models' / self.task_name / self.model_name
        
        self.model_path = self.model_dir / 'best.ckpt'
        self.processor_path = self.model_dir / 'processor.pkl'
        self.results_path = output_dir / 'results' / self.task_name / self.model_name / 'metrics.json'

    def load_or_train(self, force_train: bool=False):

        if force_train or not (self.model_path.exists() and self.processor_path.exists()):
            logger.info('Could not load existing model/processor, training a new one.')
            self.train_and_evaluate()

        self.processor = self.load_processor()
        self.model = self.load_model()

    def load_model(self):
        return self.build_model(self.config, self.model_path)

    def load_processor(self):
        return io.load_pickle(self.processor_path)

    def prepare_dataloaders(self, split_info, config):

        self.processor = self.build_processor()
        self.processor.fit(split_info.train)
        io.save_pickle(self.processor, self.processor_path)

        split_info.train = self.processor(split_info.train)
        split_info.val = self.processor(split_info.val)
        split_info.test = self.processor(split_info.test)

        collator = self.build_collator()

        train_dl = DataLoader(split_info.train, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
        val_dl = DataLoader(split_info.val, batch_size=config.batch_size, shuffle=False, collate_fn=collator)
        test_dl = DataLoader(split_info.test, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

        return train_dl, val_dl, test_dl

    def build_core_model(self, config):
        return self.model_cls(
            model_card=config.model_card,
            is_transformer_trainable=config.is_transformer_trainable,
            dropout_rate=config.dropout_rate,
            head=config.head,
        )
    
    def build_model(self, config, checkpoint_path: str = None):
        
        core_model = self.build_core_model(config)

        if checkpoint_path is not None:
            model = MAMKitLightingModel.load_from_checkpoint(
                checkpoint_path,
                model=core_model,
                loss_function=config.loss_function,
                num_classes=config.num_classes,
                optimizer_class=config.optimizer,
                val_metrics=MetricCollection(self.metric_dict),
                test_metrics=MetricCollection(self.metric_dict),
                **config.optimizer_args
            )
        else:
            model = MAMKitLightingModel(
                model=core_model,
                loss_function=config.loss_function,
                num_classes=config.num_classes,
                optimizer_class=config.optimizer,
                val_metrics=MetricCollection(self.metric_dict),
                test_metrics=MetricCollection(self.metric_dict),
                **config.optimizer_args
            )

        return model
    
    def format_pred(self, pred):
        """Override if subclasses need string labels instead of bools"""
        return bool(pred)

    def format_prob(self, prob):
        """Pick correct probability (default: positive class)."""
        return float(prob[1] if len(prob) > 1 else prob[0])

    def train_and_evaluate(self):
        seed_everything(42)

        split_key = 'mancini-et-al-2022' if self.task_name == 'arc' else 'default'
        split_info = next(iter(self.loader.get_splits(key=split_key)))
        train_dl, val_dl, test_dl = self.prepare_dataloaders(split_info, self.config)

        model = self.build_model(self.config)

        tb_logger = TensorBoardLogger(
            save_dir=str(self.model_dir.parent),
            name=self.model_dir.name,
            version="", # empty string avoids version_X
        )

        trainer = L.Trainer(
            default_root_dir=self.model_dir,
            **self.trainer_args,
            callbacks=[
                EarlyStopping(monitor='val_loss', mode='min', patience=5),
                ModelCheckpoint(
                    dirpath=self.model_dir,
                    filename='best',
                    save_top_k=1,
                    monitor='val_loss',
                    mode='min'
                ),
                PycharmProgressBar(),
            ],
            logger=tb_logger
        )

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        logger.info('Training complete.')

        val_metrics = trainer.test(ckpt_path='best', dataloaders=val_dl)[0]
        test_metrics = trainer.test(ckpt_path='best', dataloaders=test_dl)[0]
        metrics = {'validation': val_metrics, 'test': test_metrics}
        io.save_json(metrics, self.results_path)
        logger.info(f'Saved training results to {self.results_path}')

    def build_collator(self):
        return UnimodalCollator(
            features_collator=TextTransformerCollator(
                model_card=self.config.model_card,
                tokenizer_args=self.config.tokenizer_args
            ),
            label_collator=lambda labels: th.tensor(labels)
        )

    def build_processor(self):
        return UnimodalProcessor()

    def predict(self, series: list, batch_size: int = None) -> list:

        if self.task_name != 'arc': # to prevent flooding logs
            logger.info(f'Start inference...')

        sentences = [(s, 0, None) for s in series]
        processed = self.processor(sentences)

        collator = self.build_collator()
        if batch_size is None:
            batch_size = self.config.batch_size
            
        dataloader = DataLoader(processed, batch_size=batch_size, shuffle=False, collate_fn=collator)

        all_preds, all_probs = [], []

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with th.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                # Unpack tuple: (features, labels)
                inputs, _ = batch

                # Move tensors to device
                inputs = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in inputs.items()}

                # Forward pass with AMP 
                # AMP context
                if device.type == "cuda":
                    with th.amp.autocast(device_type='cuda', dtype=th.float16):
                        logits = self.model(inputs)
                else:
                    logits = self.model(inputs)

                probs = th.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

                # Log progress every N batches
                if self.task_name != 'arc' and batch_idx % 1000 == 0:
                    logger.info(f"Inference progress: batch {batch_idx}/{len(dataloader)} "
                                f"({batch_idx/len(dataloader)*100:.1f}%) "
                                f"GPU mem: {th.cuda.memory_allocated()/1024**2:.1f}MiB")

        # Map predictions back to original series
        preds = list(map(self.format_pred, all_preds))
        probs = list(map(self.format_prob, all_probs))

        if self.task_name != 'arc': # to prevent flooding logs
            logger.info('Inference complete.')

        return preds, probs
