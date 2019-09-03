import os
from dotenv import load_dotenv

load_dotenv()

model_data = os.environ.get('ENTITY_DETECTOR_MODEL_DIR')
stanford_core_nlp_dir = os.environ.get('STANFORD_CORE_NLP')
stanford_ner_dir = os.environ.get('STANFORD_NER_DIR')
mongo_id = os.environ.get('MONGODB_ID')
mongo_pass = os.environ.get('MONGODB_PASS')
mongo_host = os.environ.get('MONGODB_HOST')


class Params:
    rnn1_layers = 1
    bidirectional = False
    units1 = 256
    dropout1 = 0.5
    optimizer = "adam"
    window_size = 3
    position_emb = 3
    batch_size = 128
    gpu = False
    property2idx = os.path.join(model_data, "resources", "labellist", "property2idx.txt")
    wordembeddings = os.path.join(model_data, "resources", "glove", "glove.6B.50d.txt")
    max_sent_len = 36


class Config:
    mongolink = f"mongodb://{mongo_id}:{mongo_pass}@{mongo_host}:27017/"
    # f"mongodb://entdetectadmin:wast232word627@mongodb-756-0.cloudclusters.net:27017/"
    db = "lexnex"
    n = -1
    rawpath = model_data
    rawcoll = "entdetect_initial_text"
    stan_ner = "entdetect_stan_ner"
    spacy_ner = "entdetect_spacy_ner"
    poly_ner = "entdetect_poly_ner"
    nltk_ner = "entdetect_nltk_ner"
    sent_coll = "entdetect_sent_data"
    entgold = "entdetect_gld_withf1"
    entcurated = "curated"
    verbbydocsent = "verbdocsent"
    verb = "verb"
    topics = "topic_list"
    topic_coll = "dominant_topic"
    spacy_nc = "entdetect_noun_chunks"
    cluster_df = "cluster_id_doc"
    cluster_word_df = "cluster_word"
    crfpath = os.path.join(stanford_ner_dir, "classifiers", "english.all.3class.distsim.crf.ser.gz")
    nerjarpath = os.path.join(stanford_ner_dir, "stanford-ner.jar")


params_dict = {k: v for k, v in [(attribute, getattr(Config, attribute)) for attribute in dir(Config) if
                                 not callable(getattr(Config, attribute)) and '__' not in attribute]}
