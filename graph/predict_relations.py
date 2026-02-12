import json
import itertools 

# Read in the data
with open('neighbors.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

with open('docs.jsonl', 'r') as f:
    docs = [json.loads(line) for line in f]
    document_map = {doc['doc_id']: doc for doc in docs}


for entry in data:
    doc_id = entry['doc_id']
    neighbors = entry['neighbors']

    if doc_id not in document_map or len(neighbors) < 1:
        continue
    
    all_doc_ids = [doc_id] + neighbors
    all_docs = [document_map[d] for d in all_doc_ids if d in document_map]
    all_texts = 

    pairs = list(itertools.combinations(all_docs, 2))




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
