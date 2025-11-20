from src.argument_mining.base_model import BaseModel

from torchmetrics.classification import F1Score

from mamkit.data.datasets import MMUSED, InputMode
from mamkit.configs.text import TransformerConfig
from mamkit.models.text import Transformer


class ComponentClassifier(BaseModel):

    def __init__(self):
        self.task_key = 'component'
        self.id2label = {0: 'premise', 1: 'claim'}

        super().__init__(
            config_cls=TransformerConfig,
            model_cls=Transformer,
            model_name='roberta',
            task_name='acc',
            dataset='mmused',
            metric_dict={'f1': F1Score(task='multiclass', num_classes=2)},
            trainer_args={'accelerator': 'gpu', 'accumulate_grad_batches': 3, 'max_epochs': 20},
            num_classes=2,
            tags={'anonymous', 'roberta'}
        )

        self.config.tokenizer_args = {'truncation': True, 'max_length': 512}

        self.loader = MMUSED(
            task_name=self.task_name,
            input_mode=InputMode.TEXT_ONLY,
            base_data_path=self.mamkit_dir,
        )
    
    def format_pred(self, pred):
        return self.id2label[pred]
