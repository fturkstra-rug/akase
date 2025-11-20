from src import io_utils as io
from src.preprocessing import clean_html_fast, sentence_tokenize
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from src.argument_mining.sentence_detection import SentenceDetector
from src.argument_mining.component_classification import ComponentClassifier
from src.argument_mining.relation_classification import RelationClassifier, CustomRelationClassifier
from src.opensearch import upload_to_index
import boto3
from botocore.config import Config
import logging
logger = logging.getLogger(__name__)

def preprocess(series: pd.Series) -> pd.Series:
    logger.info(f'Start preprocessing...')

    preprocessed = (
        clean_html_fast(series.astype(str).fillna(''))
        .apply(sentence_tokenize)
    )

    logger.info(f'Preprocessing complete.')
    return preprocessed

def windowed_pairs(sentences, window=5):
    n = len(sentences)
    for i in range(n):
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if i != j:
                yield (sentences[i], sentences[j])

def run_pipeline(input_dir: str | Path):

    logger.info('Start pipeline.')

    logger.info('Loading MAMKIT models...')
    asd_model = SentenceDetector()
    asd_model.load_or_train(force_train=False)
    logger.info('Loaded SentenceDetector.')
    acc_model = ComponentClassifier()
    acc_model.load_or_train(force_train=False)
    logger.info('Loaded ComponentClassifier.')
    arc_model = CustomRelationClassifier()
    # arc_model.load_or_train(force_train=False) 
    logger.info('Loaded RelationClassifier.')

    S3_BUCKET = "bucket_name"
    S3_PREFIX = "prefix"

    config = Config(max_pool_connections=50)
    s3 = boto3.client('s3', config=config)

    for df_idx, df in tqdm(enumerate(io.load_parquet_files(input_dir))):
        
        """ TODO DEBUG """
        # if len(df) > 2000: # TODO remove, for debug
        #     print('skipping', len(df))
        #     continue
            
        # df = df[:10]
        """ TODO DEBUG """
        
        logger.info(f'Processing file {df_idx} with {len(df)} rows.')
        
        content = df['main_content'] # series of strings
        texts = preprocess(content) # series of texts, a text is a list of strings

        logger.info('Argumentative Sentence Detection...')
        # Flatten sentences but keep track of their original positions
        positions = [(doc_idx, sent_idx) for doc_idx, doc in enumerate(texts) for sent_idx, _ in enumerate(doc)]
        sentences = [sent for doc in texts for sent in doc]

        asd_preds, asd_probs = asd_model.predict(sentences)

        logger.info('Argumentative Component Classification...')
        # Filter sentences that are not argumentative
        arg_sentences = []
        arg_positions = []
        for i, s in enumerate(sentences):
            if asd_preds[i]:
                arg_sentences.append(s)
                arg_positions.append(positions[i])

        arg_position_to_idx = {pos: i for i, pos in enumerate(arg_positions)}

        if arg_sentences:
            acc_preds, acc_probs = acc_model.predict(arg_sentences)
        else:
            acc_preds, acc_probs = [], []

        # Store all predictions in a new column with sentence dictionaries
        for sentence, position, asd_pred, asd_prob in zip(sentences, positions, asd_preds, asd_probs):
            doc_idx, sent_idx = position

            # Present for all sentences
            sentence_dict = {
                'text': sentence,
                'is_arg': {'pred': asd_pred, 'prob': asd_prob},
            }

            # Component, part of sentence_dict, only present if sentence is argumentative
            if asd_pred:
                arg_idx = arg_position_to_idx[position]
                sentence_dict['arg_component'] = {'pred': acc_preds[arg_idx], 'prob': acc_probs[arg_idx]}

            texts[doc_idx][sent_idx] = sentence_dict

        logger.info('Argumentative Relation Classification...')

        all_relations = [[] for _ in range(len(texts))]

        # Get argumentative sentences per doc
        arg_positions_by_doc = defaultdict(list)

        for doc_idx, sent_idx in arg_positions:
            arg_positions_by_doc[doc_idx].append(sent_idx)

        # For each document
        for doc_idx, sent_indices in arg_positions_by_doc.items():
            # Get the argumentative sentences
            doc_sentences = [texts[doc_idx][i]['text'] for i in sent_indices]

            if len(doc_sentences) < 2:
                # Not enough argumentative units to form a relation
                continue

            # possible_pairs = permutations(doc_sentences, 2)
            possible_pairs = list(windowed_pairs(doc_sentences, window=5))
            arc_preds, arc_probs = arc_model.predict(possible_pairs)

            relations = []
            idx_map = {s: i for i, s in enumerate(doc_sentences)}

            for i, (source, target) in enumerate(possible_pairs):
                if arc_preds[i] == 'none':
                    continue
                
                relations.append({
                    'source': idx_map[source],
                    'target': idx_map[target],
                    'type': arc_preds[i],
                    'prob': arc_probs[i],
                })

            all_relations[doc_idx] = relations

        df['relations'] = all_relations
        df['sentences'] = texts

        """
        Source and target are the indices of the sentences (within the document).
        Possible type values are 'supports' or 'attacks' or neither (if neither, it is not stored).
        Possibile arg_component values are 'premise' and 'claim'.
        
        row = {
            ...,
            'main_content': <string>,
            'sentences': [
                {
                    'text': <string>,
                    'is_arg': {'pred': <bool>, 'prob': <float>},
                    'arg_component': {'pred': <string>, 'prob': <float>}
                },
                ...,
            ],
            'relations': [
                {
                    'source': <int>,
                    'target': <int>,
                    'type': <string>,
                    'prob': <float>,
                },
                ...,
            ]
        }
        """

        logger.info(f'Processing file {df_idx} with {len(df)} rows complete.')

        # Upload file to S3
        doc_ids = io.bulk_upload_to_s3(s3, df, S3_BUCKET, S3_PREFIX)
        logger.info(f'Uploaded files to s3.')
        
        # Prepare file for indexing
        upload_to_index(df)
        logger.info(f'Uploaded to index.')

    logger.info('Pipeline complete.')



