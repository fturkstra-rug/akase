from src.argument_mining.base_model import BaseModel

import torch as th
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from mamkit.configs.text import BiLSTMConfig
from mamkit.models.text import PairTransformer, PairBiLSTM
from mamkit.data.datasets import MArg, InputMode, PairUnimodalDataset
from mamkit.data.collators import PairUnimodalCollator, PairTextCollator
from mamkit.data.processing import PairUnimodalProcessor, PairVocabBuilder
from mamkit.utility.metrics import ClassSubsetMulticlassF1Score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from itertools import islice
import logging
logger = logging.getLogger(__name__)

class CustomRelationClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")
        self.model = AutoModelForSequenceClassification.from_pretrained("raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L")
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.id2label = {0: 'none', 1: 'inference', 2: 'conflict', 3: 'rephrase'}

    def format_pred(self, pred):
        return self.id2label[pred]
    
    def format_prob(self, prob):
        return prob

    def batch_iters(self, iterable, batch_size):
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch

    def predict(self, pairs, batch_size=128):
        preds, probs = [], []

        # for i in range(0, len(pairs), batch_size): # does not support generators
            # batch = pairs[i:i+batch_size]
        for batch in self.batch_iters(pairs, batch_size):
            texts = [p[0] for p in batch]
            texts2 = [p[1] for p in batch]

            inputs = self.tokenizer(
                texts, texts2,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device, non_blocking=True)

            with th.no_grad():
                with autocast():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    prob_tensor = th.softmax(logits, dim=-1)
                    pred_tensor = prob_tensor.argmax(dim=-1)

            preds.extend([self.format_pred(idx) for idx in pred_tensor.cpu().tolist()])
            probs.extend([self.format_prob(vec) for vec in prob_tensor.cpu().tolist()])

        return preds, probs


class RelationClassifier(BaseModel):

    def __init__(self):
        self.task_key = 'relation'
        self.id2label = {0: 'neither', 1: 'support', 2: 'attack'}

        super().__init__(
            config_cls=BiLSTMConfig,
            model_cls=PairTransformer,
            model_name='lstm',
            task_name='arc',
            dataset='marg',
            metric_dict={'f1': ClassSubsetMulticlassF1Score(num_classes=3, class_subset=[1, 2])},
            trainer_args={'accelerator': 'gpu', 'accumulate_grad_batches': 3, 'max_epochs': 20},
            num_classes=3,
            tags={'anonymous'}
        )

        self.loader = MArg(
            task_name=self.task_name,
            confidence=0.85,
            input_mode=InputMode.TEXT_ONLY,
            base_data_path=self.mamkit_dir
        )
    
    def format_pred(self, pred):
        return self.id2label[pred]

    def format_prob(self, prob):
        return prob
    
    def build_collator(self):
        return PairUnimodalCollator(
            features_collator=PairTextCollator(
                tokenizer=self.config.tokenizer,
                vocab=self.processor.features_processor.vocab
            ),
            label_collator=lambda labels: th.tensor(labels)
        )
    
    def build_processor(self):
        return PairUnimodalProcessor(
            features_processor=PairVocabBuilder(
                tokenizer=self.config.tokenizer,
                embedding_model=self.config.embedding_model,
                embedding_dim=self.config.embedding_dim
            )
        )

    def build_core_model(self, config):
        return PairBiLSTM(
            vocab_size=len(self.processor.features_processor.vocab),
            embedding_dim=config.embedding_dim,
            dropout_rate=config.dropout_rate,
            lstm_weights=config.lstm_weights,
            head=config.head,
            embedding_matrix=self.processor.features_processor.embedding_matrix
        )

    def predict(self, series: list):
        # series is a list of tuples
        logger.info(f'Start inference...')

        dataset = PairUnimodalDataset(
            a_inputs=[p[0] for p in series],
            b_inputs=[p[1] for p in series],
            a_context=[None] * len(series),
            b_context=[None] * len(series),
            labels=[-1] * len(series)  # dummy labels at inference
        )
        processed = self.processor(dataset)

        collator = self.build_collator()
        dataloader = DataLoader(processed, batch_size=self.config.batch_size, shuffle=False, collate_fn=collator)

        all_preds, all_probs = [], []

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        with th.no_grad():
            for batch in dataloader:
                # Unpack tuple: (features, labels)
                inputs, _ = batch

                # Move tensors to device
                inputs = {k: v.to(device) if isinstance(v, th.Tensor) else v for k, v in inputs.items()}

                # Forward pass
                logits = self.model(inputs)
                probs = th.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        # Map predictions back to original series
        preds = list(map(self.format_pred, all_preds))
        probs = list(map(self.format_prob, all_probs))

        logger.info('Inference complete.')

        return preds, probs
