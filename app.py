from flask import Flask, request, redirect, url_for, jsonify
from flask_cors import CORS
from src.parser import RelParser
from src import keras_models
from src import entity_extraction
from src import nlpent
from keras import backend as K
import json
from pycorenlp import StanfordCoreNLP
import sys
import pandas as pd
from rake_nltk import Rake

app = Flask(__name__)
app.config["DEBUG"] = False
CORS(app)
nlp_corenlp = StanfordCoreNLP('http://localhost:9000')
keras_models.model_params['wordembeddings'] = "data/resources/glove/glove.6B.50d.txt"


def getkeys(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases_with_scores()


def resolve(corenlp_output):
    """ Transfer the word form of the antecedent to its associated pronominal anaphor(s) """
    for coref in corenlp_output['corefs']:
        mentions = corenlp_output['corefs'][coref]
        antecedent = mentions[0]  # the antecedent is the first mention in the coreference chain
        for j in range(1, len(mentions)):
            mention = mentions[j]
            if mention['type'] == 'PRONOMINAL':
                # get the attributes of the target mention in the corresponding sentence
                target_sentence = mention['sentNum']
                target_token = mention['startIndex'] - 1
                # transfer the antecedent's word form to the appropriate token in the sentence
                corenlp_output['sentences'][target_sentence - 1]['tokens'][target_token]['word'] = antecedent['text']


def get_resolved(corenlp_output):
    """ Get the "resolved" output """
    out = []
    possessives = ['hers', 'his', 'their', 'theirs']
    for sentence in corenlp_output['sentences']:
        for token in sentence['tokens']:
            output_word = token['word']
            # check lemmas as well as tags for possessive pronouns in case of tagging errors
            if token['lemma'] in possessives or token['pos'] == 'PRP$':
                output_word += "'s"  # add the possessive morpheme
            output_word += token['after']
            out.append(output_word)
    return (' '.join(out))


@app.route('/predict_dl', methods=['GET', 'POST'])
def predict_dl():
    request.get_json(force=True)
    # try:
    if request.method == 'POST':
        K.clear_session()
        text = request.json.get("inputtext")
        if not text:
            tagged = entity_extraction.get_tagged_from_server("Germany is a country in Europe")
        else:
            tagged = entity_extraction.get_tagged_from_server(text)
        print(tagged)
        entity_fragments = entity_extraction.extract_entities(tagged)
        edges = entity_extraction.generate_edges(entity_fragments)
        non_parsed_graph = {'tokens': [t for t, _, _ in tagged],
                            'edgeSet': edges}
        relparser = RelParser("model_ContextWeighted", models_folder="data/trained/")
        parsed_graph = relparser.classify_graph_relations([non_parsed_graph])
        log = {}
        log['relation_graph'] = parsed_graph[0]
        K.clear_session()
        return json.dumps(log)
    else:
        return "No result"
    # except:
    #     return "No relation detected"


@app.route('/predict_openie', methods=['GET', 'POST'])
def predict_openie():
    request.get_json(force=True)
    try:
        if request.method == 'POST':
            text = request.json.get("inputtext")
            output = nlp_corenlp.annotate(text, properties={'annotators': 'dcoref', 'outputFormat': 'json',
                                                            'ner.useSUTime': 'false'})
            resolve(output)
            coreftext = get_resolved(output)
            output = nlp_corenlp.annotate(coreftext, properties={'annotators': 'openie', 'outputFormat': 'json',
                                                                 'ner.useSUTime': 'false'})
            out = list(map(lambda x: x['openie'], output['sentences']))
            flat_list = [item for sublist in out for item in sublist]
            dfnc, entdf, desiglist = nlpent.ner(coreftext)
            dfnc = dfnc.to_dict(orient="records")
            entdf = entdf.to_dict(orient="records")
            keyconcepts = pd.DataFrame(list(getkeys(text)))
            keyconcepts.columns = ["Score", "Keywords"]
            keyconcepts = keyconcepts.to_dict(orient="records")
            return jsonify({"ie": flat_list, "dfnc": dfnc, "entdf": entdf, "kc": keyconcepts, "ds": desiglist})
        else:
            return "No result"
    except Exception as e:
        print(str(e))
        return jsonify({"ie": "No relation detected"})


#
# @app.route('/startPreprocess',methods = ['GET'])
# def startPreprocess():
#     app = entdetect.AsyncNLPProcess()
#     app.run()
#     return "Started Preprocessing"


@app.route('/startTrain', methods=['GET'])
def startTrain():
    return 'Hello World!'


@app.route("/")
def hello():
    return "Welcome to ER Detector"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
