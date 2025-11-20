# Argmining Pipeline

1. Preprocess web documents.
2. Detect argumentative sentences.
3. Classify the argumentative sentences as either a claim or premise.
4. Predict relations between argumentative sentences (intra-document only) using a sliding window approach.
5. Store the enriched web document data in AWS S3.
6. Index the new documents with AWS OpenSearch.
