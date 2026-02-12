# Testing Argument Mining Models on Web Data Benchmark

## Benchmark

Ivan Habernal, Iryna Gurevych; Argumentation Mining in User-Generated Web Discourse. Computational Linguistics 2017; 43 (1): 125–179. doi: https://doi.org/10.1162/COLI_a_00276

Download link: https://tudatalib.ulb.tu-darmstadt.de/items/d4a7ac0c-e7a8-466a-855a-917e42a24342

## Tasks

- Argumentative Sentence Detection (boolean).
- Argumentative Component Classification (claim | premise).
- Argumentative Relation Classification (inference | conflict | rephrase | none).

Notes: We equate 'persuasiveness' with 'argumentative' here. We ignore the other three components present in the benchmark (backing, rebuttal, refutation). We infer relation types as follows:

Premise --> Claim = support
Backing --> Claim = support
Backing --> Premise = support
Rebuttal --> Claim = attack
Refutation --> Rebuttal = attack

## Models

- asd_roberta_mmused_text_only (MAMKIT)
- acc_roberta_marg_text_only (MAMKIT)
- raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L (HuggingFace)

Eleonora Mancini, Federico Ruggeri, Stefano Colamonaco, Andrea Zecca, Samuele Marro, Paolo Torroni; MAMKit: A Comprehensive Multimodal Argument Mining Toolkit. Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024), 2024; 69–82. doi: https://doi.org/10.18653/v1/2024.argmining-1.7

Github: https://github.com/nlp-unibo/mamkit

Ramon Ruiz-Dolz, Jose Alemany, Stella M. Heras Barbera, Ana Garcia-Fornes; Transformer-Based Models for Automatic Identification of Argument Relations: A Cross-Domain Evaluation. IEEE Intelligent Systems, 2021; 36(6): 62–70. doi: https://doi.org/10.1109/mis.2021.3073993

HuggingFace: https://huggingface.co/raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L
