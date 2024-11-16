import json
import os
import datasets
from enum import Enum
import datasets
import nltk
from datasets import DatasetDict, load_dataset
nltk.download('punkt_tab')

class Tasks(Enum):
    # Knowledge Base
    DEPENDENCY_PARSING = "DEP"
    KEYWORD_EXTRACTION = "KE"
    WORD_ANALOGY = "WA"
    WORD_SENSE_DISAMBIGUATION = "WSD"
    COREFERENCE_RESOLUTION = "COREF"
    RELATION_EXTRACTION = "RE"

    # Tree
    CONSTITUENCY_PARSING = "CONST_PAR"

    # Single Text Classification (single-label)
    ABUSIVE_LANGUAGE_PREDICTION = "ABL"
    COMPLAINT_DETECTION = "CD"
    DOMAIN_KNOWLEDGE_CLASSIFICATION = "DKC" # classification for non NLP-oriented label
    EMOTION_CLASSIFICATION = "EC"
    LANGUAGE_IDENTIFICATION = "LI"
    HOAX_NEWS_CLASSIFICATION = "HNC"
    INTENT_CLASSIFICATION = "INT"
    LEGAL_CLASSIFICATION = "LC"
    MORALITY_CLASSIFICATION = "MC"
    READABILITY_ASSESSMENT = "RA"
    RHETORIC_MODE_CLASSIFICATION = "RMC"
    SENTIMENT_ANALYSIS = "SA"
    TAX_COURT_VERDICT = "TACOS"
    TOPIC_MODELING = "TL"
    REINFORCEMENT_LEARNING_WITH_HUMAN_FEEDBACK = "RLHF"

    # Single Text Classification (multi-label)
    ASPECT_BASED_SENTIMENT_ANALYSIS = "ABSA"
    DOMAIN_KNOWLEDGE_MULTICLASSIFICATION = "DKM" # multi-classification for non NLP-oriented label
    CODE_SWITCHING_IDENTIFICATION = "CSI"

    # Single Text Sequence Labeling
    KEYWORD_TAGGING = "KT"
    NAMED_ENTITY_RECOGNITION = "NER"
    POS_TAGGING = "POS"
    SENTENCE_ORDERING = "SO"
    SLOT_FILLING = "SF"
    SPAN_BASED_ABSA = "SPAN_ABSA"
    TOKEN_LEVEL_LANGUAGE_IDENTIFICATION = "LANGID"

    # Pair Text Classification
    COMMONSENSE_REASONING = "CR"
    QUESTION_ANSWERING = "QA"
    QUESTION_ANSWERING_RETRIEVAL = "QAR"
    TEXT_RETRIEVAL = "TRV"
    TEXTUAL_ENTAILMENT = "TE"
    SEMANTIC_SIMILARITY = "STS"
    NEXT_SENTENCE_PREDICTION = "NSP"
    SHORT_ANSWER_GRADING = "SAG"
    MORPHOLOGICAL_INFLECTION = "MOR"
    CONCEPT_ALIGNMENT_CLASSIFICATION = "CAC"

    # Single Text Generation
    CROSS_LINGUAL_SUMMARIZATION = "X-SUM"
    INSTRUCTION_TUNING = "ITT"
    MACHINE_TRANSLATION = "MT"
    MULTILEXNORM = "MLN"
    PARAPHRASING = "PARA"
    SUMMARIZATION = "SUM"
    TRANSLITERATION = "TRL"

    # Multi Text Generation
    DIALOGUE_SYSTEM = "DS"
    E2E_TASK_ORIENTED_DIALOGUE = "TOD"
    MULTI_TURN_CONVERSATION = "MTC"

    # Self Supervised & Unsupervised Text
    PROMPTING = "PRT"
    SELF_SUPERVISED_PRETRAINING = "SSP"

    # SpeechText
    SPEECH_RECOGNITION = "ASR"
    SPEECH_TO_TEXT_TRANSLATION = "STTT"

    SPEECH_LANGUAGE_IDENTIFICATION = "SPEECH_LID"
    SPEECH_EMOTION_RECOGNITION = "SER"
    SPEECH_EMOTION_RECOGNITION_MULTILABEL = "SER_MULTI"

    TEXT_TO_SPEECH = "TTS"

    # SpeechSpeech
    SPEECH_TO_SPEECH_TRANSLATION = "S2ST"

    # Image
    IMAGE_CLASSIFICATION = "IMC"
    IMAGE_CLASSIFICATION_MULTILABEL = "IMC_MULTI"

    # ImageText
    IMAGE_CAPTIONING = "IC"
    VISUAL_QUESTION_ANSWERING = "VQA"
    SIGN_LANGUAGE_RECOGNITION = "SLR"
    STYLIZED_IMAGE_CAPTIONING = "SIC"
    VISUALLY_GROUNDED_REASONING = "VGR"
    OPTICAL_CHARACTER_RECOGNITION = "OCR"

    # VideoText
    VIDEO_CAPTIONING = "VC"
    VIDEO_TO_TEXT_RETRIEVAL = "V2TR"

    # No seacrowd schema
    FACT_CHECKING = "FCT"
    WORD_LIST = "WL"

def patch_resolve_trust_remote_code():
    def resolve_trust_remote_code(trust_remote_code: bool | None, repo_id: str):
        print('Patch `trust_remote_code` to enable fully auto-run. Beware of the risk of code injection in the dataset.')
        return True
    datasets.load.resolve_trust_remote_code = resolve_trust_remote_code

patch_resolve_trust_remote_code()


NLU_TASK_LIST = [
    "wisesight_thai_sentiment_seacrowd_text",
    "m3exam_tha_seacrowd_qa",
    "xcopa_tha_seacrowd_qa",
    "belebele_tha_thai_seacrowd_qa",
    "xnli.tha_seacrowd_pairs",
    'thaiexam_qa'
]


NLG_TASK_LIST = [
    "xl_sum_tha_seacrowd_t2t",
    "flores200_eng_Latn_tha_Thai_seacrowd_t2t",
    "flores200_tha_Thai_eng_Latn_seacrowd_t2t",
    "iapp_squad_seacrowd_qa",
]

def _get_task_from_value(value: str):
    for task_cls in Tasks:
        if task_cls.value == value:
            return task_cls
    raise ValueError(f"No matching enum found for value: {value}")


def create_dataset_config(row):
    dataset_name, task_value, dataset_config  = row
    task_cls = _get_task_from_value(task_value)
    if dataset_config['use_file']:
        ds = DatasetDict()
        for s in dataset_config['subset']:
            d = load_dataset(dataset_config['repo'], data_files=f'{dataset_name}_{s}.jsonl')['train']
            ds[s] = d
    else:
        ds = load_dataset(dataset_config['repo'])
    return (ds, task_cls)

def dataset_from_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}
    with open(f'{current_dir}/config/dataset_config.json') as f:
        dataset_config = json.load(f)
        for config_name in dataset_config.keys():
            results[config_name] = create_dataset_config(dataset_config[config_name])
    return results
    


def load_nlu_datasets():
    dataset = dataset_from_config()
    cfg_name_to_dset_map = {}
    for config_name in NLU_TASK_LIST:
        cfg_name_to_dset_map[config_name] = dataset[config_name]
    return cfg_name_to_dset_map


def load_nlg_datasets():
    dataset = dataset_from_config()
    cfg_name_to_dset_map = {}
    for config_name in NLG_TASK_LIST:
        cfg_name_to_dset_map[config_name] = dataset[config_name]
    return cfg_name_to_dset_map