from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    method: str = field(default=384)

    # Seq2Seq model specific args
    generation_max_len: int = field(default=60)
    generation_beam_size: int = field(default=10)
    generation_do_sample: bool = field(default=False)
    generation_length_penalty: float = field(default=1.0)
    generation_uid_regularization: float = field(default=0.0)
    generation_no_repeat_ngram_size: int = field(default=3)
    generation_k_2: int = field(default=0)
    generation_typical_p: float = field(default=1.0)

    # Tokenization
    history_max_tokens: int = field(default=384)
    history_max_utterances: int = field(default=999)
    knowledge_max_tokens: int = field(default=128)

    # Special Tokens
    user_token: str = field(default="<user>")
    agent_token: str = field(default="<agent>")
    knowledge_sep_token: str = field(default="<knowledge_sep>")
    knowledge_tag_token: str = field(default="<knowledge_tag>")

    # Cross encoder
    selection_level: str = field(default='all',
        metadata={"help": "Which level of the knowledge snippet should be selected. Can be one of [all, document, entity, domain, domain_entity]"})
    num_domain_negatives: int = field(default=1,
        metadata={"help": "Number of negative knowledge snippets with a different domain."})
    num_entity_negatives: int = field(default=1,
        metadata={"help": "Number of negative knowledge snippets with a different entity but the same domain."})
    num_doc_negatives: int = field(default=1,
        metadata={"help": "Number of negative knowledge snippets with a different document but the same entity."})
    sample_document_uniform: bool = field(default=False)
    sample_dialog_contexts: bool = field(default=False)

    selection_prediction_topk: int = field(default=20,
        metadata={"help": "Which level of the knowledge snippet should be selected. Can be one of [all, document, entity, domain, domain_entity]"})

    hierarchical_selection_beam_search: bool = field(default=False)
    hierarchical_selection_beam_threshold: float = field(default=0.95,
        metadata={"help": "If beam search is used in the hierarchical selection consider all hypothesis above the threshold"})
    hierarchical_selection_beam_factor: float = field(default=1.0)

    # Selection reranking
    selection_reranking_topk: int = field(default=5)
    selection_reranking_include_non_ks: bool = field(default=True)
    selection_reranking_model_type: str = field(default='seq_class',
        metadata={"help": "`seq_class`: Use CLS token with topk classes, `multiple_choice` linear layer on top of the individual start tokens"})

    # Bi bencoder
    bi_encoder_loss: str = field(default="triplet")
    bi_encoder_shared: bool = field(default=False,
                                    metadata={"help": "Share the parameters of both query and document encoder."})

    triplet_loss_margin: float = field(default=5.0)

    process_all_in_nbest_last_turn: bool = field(default=False,
                                                 metadata={
                                                     "help": "Run the model on all hyps in the nbest list."})
    detection_nbest_combination_stratey: str = field(default='max')
    selection_nbest_combination_stratey: str = field(default='weighted_by_asr')

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None, 
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_filter_dict: Optional[dict] = field(
        default=None, 
    )
    dataset_train_split: Optional[str] = field(default="train")
    dataset_eval_split: Optional[str] = field(default=None)

    is_training: bool = field(default=True)

@dataclass
class DataPredictionArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_data_files: Optional[dict] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_filter_dict: Optional[dict] = field(
        default=None, 
    )
    dataset_transformations: Optional[List[str]] = field(
        default=None, 
    )
    dataset_lowercase_entities: bool = field(default=False)
    dataset_test_split: str = field(default="test")

    test_documents_faiss_index_path: str = field(default=None)

    prediction_output_file: Optional[str] = field(default=None)

    perplexity_labels: Optional[str] = field(default=None)

    is_training: bool = field(default=False)

    metric_output_file: Optional[str] = field(default=None)
