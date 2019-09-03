import os
import re

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from polyglot.text import Text
from pycorenlp import StanfordCoreNLP

import config

tok = WordPunctTokenizer()
sent_detector = nltk.tokenize.punkt.PunktSentenceTokenizer()
nlp = spacy.load('en_core_web_sm')
nlp_spacy = spacy.load('en_core_web_sm')
nlp_corenlp = StanfordCoreNLP('http://localhost:9000')
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def tok_text(text):
    lower_case = text.lower()
    words = tok.tokenize(lower_case)
    sentences = " ".join(words)
    return sentences


def stanford_ner(txt):
    classified_text = []
    st = StanfordNERTagger(
        os.path.join(config.stanford_ner_dir, 'classifiers', 'english.all.3class.distsim.crf.ser.gz'),
        os.path.join(config.stanford_ner_dir, 'stanford-ner.jar'))
    tokenized_text = word_tokenize(txt)
    classified_text.append(st.tag(tokenized_text))
    return classified_text


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


def stan_core(text):
    try:
        output = nlp_corenlp.annotate(text, properties={'annotators': 'dcoref', 'outputFormat': 'json',
                                                        'ner.useSUTime': 'false'})
        resolve(output)
        sent_no_articles = get_resolved(output)
        sent_no_articles
    except:
        sent_no_articles = text
    return sent_no_articles


def stan_sub_list(nlist):
    allpers = []
    for k in range(0, len(nlist)):
        testlist = nlist[k]
        personlist = []
        tempp = ""
        for i in range(0, len(testlist)):
            if testlist[i][1] == "PERSON":
                tempp = tempp + " " + testlist[i][0]
            else:
                if (i >= 1 and testlist[i - 1][1] == "PERSON"):
                    personlist.append(tempp)
                    tempp = ""
        allpers.append(personlist)
    allloc = []
    for k in range(0, len(nlist)):
        testlist = nlist[k]
        loclist = []
        tempp = ""
        for i in range(0, len(testlist)):
            if testlist[i][1] == "LOCATION":
                tempp = tempp + " " + testlist[i][0]
            else:
                if (i >= 1 and testlist[i - 1][1] == "LOCATION"):
                    loclist.append(tempp)
                    tempp = ""
        allloc.append(loclist)
    allorg = []
    for k in range(0, len(nlist)):
        testlist = nlist[k]
        orglist = []
        tempp = ""
        for i in range(0, len(testlist)):
            if testlist[i][1] == "ORGANIZATION":
                tempp = tempp + " " + testlist[i][0]
            else:
                if (i >= 1 and testlist[i - 1][1] == "ORGANIZATION"):
                    orglist.append(tempp)
                    tempp = ""
        allorg.append(orglist)
    return allpers, allloc, allorg


def stan_final(l1, l2, l3):
    df_stan_ner = []
    p = l1
    l = l2
    o = l3
    if (len(l) > 0):
        for i in range(0, len(l)):
            tempdict = {}
            tempdict['ent'] = l[i]
            tempdict["label_stan"] = "LOCATION"
            df_stan_ner.append(tempdict)
    if (len(o) > 0):
        for i in range(0, len(o)):
            tempdict = {}
            tempdict['ent'] = o[i]
            tempdict["label_stan"] = "ORG"
            df_stan_ner.append(tempdict)
    if (len(p) > 0):
        for i in range(0, len(p)):
            tempdict = {}
            tempdict['ent'] = p[i]
            tempdict["label_stan"] = "PERSON"
            df_stan_ner.append(tempdict)
    df_stan_ner = pd.DataFrame.from_dict(df_stan_ner)
    return df_stan_ner


def nltk_ner(txt):
    entlist = []
    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                if chunk.label() == 'PERSON' or chunk.label() == 'GPE' or chunk.label() == 'ORGANIZATION':
                    tempdict = {}
                    tempdict['ent'] = ' '.join([c[0] for c in chunk])
                    tempdict['label_nltk'] = chunk.label()
                    entlist.append(tempdict)
    return entlist


def text_spacy_ner(txt):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp_spacy(txt)
    entlist = []
    for ent in doc.ents:
        if ent.label_ == 'ORG' or ent.label_ == 'PERSON' or ent.label_ == 'GPE':
            tempdict = {}
            tempdict['ent'] = ent.text
            tempdict['label_spacy'] = ent.label_
            entlist.append(tempdict)
    return entlist


def polyglot_ner(txt):
    text = Text(txt, hint_language_code='en')
    entlist = []
    for ent in text.entities:
        tempdict = {}
        tempdict['ent'] = ent[0]
        tempdict['label_poly'] = ent.tag
        entlist.append(tempdict)
    return entlist


def clean_text(txt):
    try:
        txt = re.sub(r'\r?\n|\r', '', txt)
        txt = re.sub(r'https\S+', '', txt, flags=re.MULTILINE)
        txt = txt.replace('\'', '')
        txt = re.sub('\S*@\S*\s?', '', txt)
        txt = re.sub(' +', ' ', txt)
        # txt = txt.replace('.','')
        txt = stan_core(txt)
    except:
        return (txt)
    return str(txt)


def create_sents(txt):
    ndf = pd.DataFrame()
    sentences = sent_detector.tokenize(txt)
    ndf['sentences'] = sentences
    return ndf


def text_spacy_nc(txt):
    doc = nlp_spacy(txt)
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    for token_nc in doc.noun_chunks:
        # if token_nc.root.dep_ == "nsubj" or token_nc.root.dep_ == "dobj":
        #     tempdict = {}
        #     tempdict['docid'] = name
        #     tempdict['Subject'] = token_nc.text
        #     tempdict['Verb'] = token_nc.label_
        #     tempdict['sent_id'] = idx
        #     tempdict['sentence'] = tok_text(txt)
        #     entlist.append(tempdict)
        x1.append(token_nc.text)
        x2.append(token_nc.root.text)
        x3.append(token_nc.root.dep_)
        x4.append(token_nc.root.head.text)
    ncdf = pd.DataFrame(
        {'Text': x1, 'root.text': x2, 'root.dep_': x3, 'root.head.text': x4})
    ncdf = ncdf[(ncdf['root.dep_'] == "nsubj") | (ncdf['root.dep_'] == "dobj")]
    ncdf['Object'] = ncdf['Text'].shift(-1)
    ncdf['root.head.text'] = ncdf['root.head.text'].shift(-1)
    ncdf = ncdf.rename(columns={'Text': 'Subject', 'root.head.text': 'Verb'})
    ncdf = ncdf[['Subject', 'Verb', 'Object']]
    ncdf.drop(ncdf.tail(1).index, inplace=True)
    ncdf_final = ncdf.to_dict(orient="records")
    return ncdf_final


def df_create_for_merge(df1, df2, df3, df4):
    df1_final = df1[['ent', 'label_stan']]
    df2_final = df2[['ent', 'label_nltk']]
    df3_final = df3[['ent', 'label_spacy']]
    df4_final = df4[['ent', 'label_poly']]
    return df1_final, df2_final, df3_final, df4_final


def titledetect(text):
    title = pd.read_csv(os.path.join(config.model_data, 'resources', 'titles_combined.txt'), sep="\t", header=None)
    title = list(title[0])
    return title


def ner(text):
    stan_core_text = clean_text(text)
    sents = create_sents(stan_core_text)
    stanner = stanford_ner(stan_core_text)
    stanl1, stanl2, stanl3 = stan_sub_list(stanner)
    stanl1 = [item for sublist in stanl1 for item in sublist]
    stanl2 = [item for sublist in stanl2 for item in sublist]
    stanl3 = [item for sublist in stanl3 for item in sublist]
    stan_ner_df_final = stan_final(stanl1, stanl2, stanl3)
    stan_ner_df_final['ent'] = list(map(lambda x: x.strip(), list(stan_ner_df_final['ent'])))
    df_spacy_ner = text_spacy_ner(stan_core_text)
    df_spacy_ner = pd.DataFrame.from_dict(df_spacy_ner)
    df_spacy_ner['ent'] = list(map(lambda x: x.strip(), list(df_spacy_ner['ent'])))
    df_poly_ner = polyglot_ner(stan_core_text)
    # df_poly_ner = [item for sublist in df_poly_ner for item in sublist]
    df_poly_ner = pd.DataFrame.from_dict(df_poly_ner)
    df_poly_ner['ent'] = list(map(lambda x: x.strip(), list(df_poly_ner['ent'])))
    df_nltk_ner = nltk_ner(stan_core_text)
    # df_nltk_ner = [item for sublist in df_nltk_ner for item in sublist]
    df_nltk_ner = pd.DataFrame.from_dict(df_nltk_ner)
    df_nltk_ner['ent'] = list(map(lambda x: x.strip(), list(df_nltk_ner['ent'])))
    df_spacy_nc = text_spacy_nc(stan_core_text)
    # df_spacy_nc = [item for sublist in df_spacy_nc for item in sublist]
    df_spacy_nc_2 = pd.DataFrame.from_dict(df_spacy_nc)
    nerstan_mer, nernltk_mer, nerspacy_mer, nerpoly_mer = df_create_for_merge(stan_ner_df_final, df_nltk_ner,
                                                                              df_spacy_ner, df_poly_ner)
    if all([len(nerstan_mer) > 0, len(nerspacy_mer) > 0]):
        ner_common_1 = pd.merge(nerstan_mer, nerspacy_mer, how='outer', on=['ent'])
    elif any([len(nerstan_mer) > 0, len(nerspacy_mer) > 0]):
        ner_common_1 = nerstan_mer if len(nerspacy_mer) == 0 else nerspacy_mer
    else:
        ner_common_1 = pd.DataFrame()
    if all([len(nerpoly_mer) > 0, len(nernltk_mer) > 0]):
        ner_common_2 = pd.merge(nerpoly_mer, nernltk_mer, how='outer', on=['ent'])
    elif any([len(nerpoly_mer) > 0, len(nernltk_mer) > 0]):
        ner_common_2 = nerpoly_mer if len(nernltk_mer) == 0 else nernltk_mer
    else:
        ner_common_2 = pd.DataFrame()
    if all([len(ner_common_1) > 0, len(ner_common_2) > 0]):
        ner_common_final_df = pd.merge(ner_common_1, ner_common_2, how='outer', on=['ent'])
    elif any([len(ner_common_1) > 0, len(ner_common_2) > 0]):
        ner_common_final_df = ner_common_1 if len(ner_common_2) == 0 else ner_common_2
    else:
        ner_common_final_df = pd.DataFrame()
    ner_common_final_df = ner_common_final_df.drop_duplicates()
    map_df = pd.read_csv(os.path.join(config.model_data, 'mapping', 'entmapping.csv'))
    map_df_spacy = map_df.rename(columns={'Initial': 'label_spacy'})
    map_df_nltk = map_df.rename(columns={'Initial': 'label_nltk'})
    map_df_poly = map_df.rename(columns={'Initial': 'label_poly'})
    ner_common_final_df_1 = pd.merge(ner_common_final_df, map_df_spacy, how='left', on=['label_spacy'])
    ner_common_final_df_1 = ner_common_final_df_1[['ent', 'label_stan', 'label_nltk', 'label_poly', 'Final']]
    ner_common_final_df_1 = ner_common_final_df_1.rename(columns={'Final': 'label_spacy'})
    ner_common_final_df_1 = pd.merge(ner_common_final_df_1, map_df_nltk, how='left', on=['label_nltk'])
    ner_common_final_df_1 = ner_common_final_df_1[['ent', 'label_stan', 'label_spacy', 'label_poly', 'Final']]
    ner_common_final_df_1 = ner_common_final_df_1.rename(columns={'Final': 'label_nltk'})
    ner_common_final_df_1 = pd.merge(ner_common_final_df_1, map_df_poly, how='left', on=['label_poly'])
    ner_common_final_df_1 = ner_common_final_df_1[['ent', 'label_stan', 'label_nltk', 'label_spacy', 'Final']]
    ner_common_final_df_1 = ner_common_final_df_1.rename(columns={'Final': 'label_poly'})
    ner_common_final_df_1 = ner_common_final_df_1.drop_duplicates()
    ner_common_final_df_1['Label'] = \
        ner_common_final_df_1[list(ner_common_final_df_1.columns)[1:]].mode(axis=1, numeric_only=False, dropna=True)[
            [0]]
    ner_common_final_df_1 = ner_common_final_df_1[['ent', 'Label']]
    ner_common_final_df_1 = ner_common_final_df_1.drop_duplicates(['ent'])
    ner_common_final_df_1['is_alphanum'] = list(
        map(lambda x: re.findall('(\w+|\s\w+)', x), list(ner_common_final_df_1['ent'])))
    ner_common_final_df_1['is_alphanum'] = list(map(lambda x: len(x) > 0, list(ner_common_final_df_1['is_alphanum'])))
    ner_common_final_df_1.dropna(axis=0, inplace=True)
    desiglist = create_list(text)
    return df_spacy_nc_2, ner_common_final_df_1, desiglist


def check(string, sub_str):
    if (string.find(sub_str) == -1):
        return "No match found!"
    else:
        return sub_str


def create_list(full_text):
    job_list = pd.read_csv(os.path.join(config.model_data, 'resources', 'titles.txt'), sep="\t", header=None)
    job_list = list(set(list(job_list[0])))
    regstring = re.compile("|".join(job_list))
    match_list = re.findall(regstring, full_text)
    return match_list
