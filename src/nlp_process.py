import glob
from polyglot.text import Text
import nltk
import re
import spacy
from pycorenlp import StanfordCoreNLP
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pymongo
from tqdm import tqdm
import json
from joblib import Parallel, delayed
from fuzzywuzzy import process
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer

tok = WordPunctTokenizer()
sent_detector = nltk.tokenize.punkt.PunktSentenceTokenizer()


nlp = spacy.load('en_core_web_sm')
nlp_spacy = spacy.load('en_core_web_sm')
nlp_corenlp = StanfordCoreNLP('http://localhost:9000')
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def read_files(path, n):
    file_list_bw = glob.glob(path + "/**/*.json", recursive=True)
    data_tx = []
    for file_path in tqdm(file_list_bw):
        with open(file_path) as json_file:
            data = json.load(json_file)
        json_file.close()
        for item in data['articles']:
            item['file_path'] = file_path
            data_tx.append(item)
    df_tx = pd.DataFrame.from_dict(data_tx)
    if (n>0):
        df_tx = df_tx.head(n)
    df_tx = df_tx.rename(columns={"id": "doc_id", "content": "text"})
    return df_tx


def stanford_ner(argsdict,txt):
    classified_text = []
    st = StanfordNERTagger(argsdict['crfpath'],argsdict['nerjarpath'], encoding='utf-8')
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
        return(txt)
    return str(txt)


def split_text(df):
    x1 = []
    x2 = []
    for i in tqdm(range(0, len(df))):
        x1_i = df.text[i].split("\n\n")
        filepath_i = df.filepath[i]
        for j in range(0, len(x1_i)):
            x1.append(x1_i[j])
            x2.append(filepath_i)
    return_df = pd.DataFrame({'file_path': x2, 'text': x1})
    return return_df


def create_sents(df):
    for i in tqdm(range(0, len(df))):
        ndf = pd.DataFrame()
        try:
            sentences = sent_detector.tokenize(df.cleaned_coref_text[i])
        except:
            continue
        ndf['sentences'] = sentences
        ndf['docid'] = df.doc_id[i]
        if (i == 0):
            sentdf = ndf
        else:
            sentdf = sentdf.append(ndf, ignore_index=True)
    return sentdf


def stan_final(l1, l2, l3, obj0):
    df_stan_ner = []
    for k in tqdm(range(0, len(l1))):
        l = l1[k]
        p = l2[k]
        o = l3[k]
        if (len(l) > 0):
            for i in range(0, len(l)):
                tempdict = {}
                tempdict['docid'] = obj0.docid.loc[k]
                tempdict['ent'] = l[i]
                tempdict["label_stan"] = "LOCATION"
                tempdict["sent_id"] = obj0.sent_id.loc[k]
                tempdict["sentence"] = tok_text(obj0.sentences.loc[k])
                df_stan_ner.append(tempdict)
        if (len(o) > 0):
            for i in range(0, len(o)):
                tempdict = {}
                tempdict['docid'] = obj0.docid.loc[k]
                tempdict['ent'] = o[i]
                tempdict["label_stan"] = "ORG"
                tempdict["sent_id"] = obj0.sent_id.loc[k]
                tempdict["sentence"] = tok_text(obj0.sentences.loc[k])
                df_stan_ner.append(tempdict)
        if (len(p) > 0):
            for i in range(0, len(p)):
                tempdict = {}
                tempdict['docid'] = obj0.docid.loc[k]
                tempdict['ent'] = p[i]
                tempdict["label_stan"] = "PERSON"
                tempdict["sent_id"] = obj0.sent_id.loc[k]
                tempdict["sentence"] = tok_text(obj0.sentences.loc[k])
                df_stan_ner.append(tempdict)
    df_stan_ner = pd.DataFrame.from_dict(df_stan_ner)
    return df_stan_ner


def nltk_ner(txt, name, idx):
    entlist = []
    for sent in nltk.sent_tokenize(txt):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label'):
                tempdict = {}
                tempdict['docid'] = name
                tempdict['ent'] = ' '.join([c[0] for c in chunk])
                tempdict['label_nltk'] = chunk.label()
                tempdict['sentence'] = tok_text(txt)
                tempdict['sent_id'] = idx
                entlist.append(tempdict)
    return entlist


def text_spacy_ner(txt, name, idx):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp_spacy(txt)
    entlist = []
    for ent in doc.ents:
        if ent.label_ == 'ORG' or ent.label_ == 'PERSON' or ent.label_ == 'GPE':
            tempdict = {}
            tempdict['docid'] = name
            tempdict['ent'] = ent.text
            tempdict['label_spacy'] = ent.label_
            tempdict['sent_id'] = idx
            tempdict['sentence'] = tok_text(txt)
            entlist.append(tempdict)
    return entlist


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def text_spacy_nc(txt, name, idx):
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
        {'Text': x1, 'root.text': x2, 'root.dep_': x3, 'root.head.text': x4, 'docid': name, 'sent_id': idx})
    ncdf = ncdf[(ncdf['root.dep_'] == "nsubj") | (ncdf['root.dep_'] == "dobj")]
    ncdf['Object'] = ncdf['Text'].shift(-1)
    ncdf['root.head.text'] = ncdf['root.head.text'].shift(-1)
    ncdf['sentence'] = tok_text(txt)
    ncdf = ncdf.rename(columns={'Text': 'Subject', 'root.head.text': 'Verb'})
    ncdf = ncdf[['Subject', 'Verb', 'Object', 'docid', 'sent_id', 'sentence']]
    ncdf.drop(ncdf.tail(1).index, inplace=True)
    ncdf_final = ncdf.to_dict(orient="records")
    return ncdf_final


def mongowrite(argsdict, collname, df):
    client = pymongo.MongoClient(argsdict['mongolink'])
    db = client[argsdict['db']]
    col0 = db[collname]
    try:
        mdict = df.to_dict(orient='records')
        col0.insert_many(mdict)
    except Exception as e:
        col0.insert_many(df)
        print(str(e))
    pass


def tok_text(text):
    lower_case = text.lower()
    words = tok.tokenize(lower_case)
    sentences = " ".join(words)
    return sentences


def polyglot(txt, name, idx):
    text = Text(txt, hint_language_code='en')
    entlist = []
    for ent in text.entities:
        tempdict = {}
        tempdict['docid'] = name
        tempdict['ent'] = ent[0]
        tempdict['label_poly'] = ent.tag
        tempdict['sent_id'] = idx
        tempdict['sentence'] = tok_text(txt)
        entlist.append(tempdict)
    return entlist


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp_spacy(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def df_create_for_merge(df1, df2, df3, df4):
    df1_final = df1[['docid', 'sent_id', 'ent', 'label_stan']]
    df2_final = df2[['docid', 'sent_id', 'ent', 'label_nltk']]
    df3_final = df3[['docid', 'sent_id', 'ent', 'label_spacy']]
    df4_final = df4[['docid', 'sent_id', 'ent', 'label_poly']]
    return df1_final, df2_final, df3_final, df4_final


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def main(argsdict):
    # read raw data
    rawdata = read_files(argsdict['rawpath'],argsdict['n'])
    filenames = list(rawdata.doc_id)
    # clean and coreference resolution
    stan_core_list = list(map(lambda x: clean_text(x), list(rawdata.text)))
    rawdata['cleaned_coref_text'] = stan_core_list
    mongowrite(argsdict, argsdict['rawcoll'], rawdata)
    # upload rawdata to database
    # create sentences
    sents = create_sents(rawdata)
    sents['sent_id'] = sents.groupby('docid').cumcount() + 1
    mongowrite(argsdict, argsdict['sent_coll'], sents)
    # create Stanford NER
    stanner = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(stanford_ner)(argsdict,sents.sentences[i]) for i in tqdm(range(0, len(sents))))
    stanl1, stanl2, stanl3 = stan_sub_list(list(map(lambda x: x[0], stanner)))
    stan_ner_df_final = stan_final(stanl1, stanl2, stanl3, sents)
    mongowrite(argsdict, argsdict['stan_ner'], stan_ner_df_final)
    # create spacy ner
    df_spacy_ner = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(text_spacy_ner)(sents.sentences[i], sents.docid[i], sents.sent_id[i]) for i in
        tqdm(range(0, len(sents))))
    df_spacy_ner = [item for sublist in df_spacy_ner for item in sublist]
    df_spacy_ner = pd.DataFrame.from_dict(df_spacy_ner)
    mongowrite(argsdict, argsdict['spacy_ner'], df_spacy_ner)
    # Get polyglot NER
    df_poly_ner = Parallel(n_jobs=8)(
        delayed(polyglot)(sents.sentences[i], sents.docid[i], sents.sent_id[i]) for i in tqdm(range(0, len(sents))))
    df_poly_ner = [item for sublist in df_poly_ner for item in sublist]
    df_poly_ner = pd.DataFrame.from_dict(df_poly_ner)
    mongowrite(argsdict, argsdict['poly_ner'], df_poly_ner)
    # get nltk ner
    df_nltk_ner = Parallel(n_jobs=8)(
        delayed(nltk_ner)(sents.sentences[i], sents.docid[i], sents.sent_id[i]) for i in tqdm(range(0, len(sents))))
    df_nltk_ner = [item for sublist in df_nltk_ner for item in sublist]
    df_nltk_ner = pd.DataFrame.from_dict(df_nltk_ner)
    mongowrite(argsdict, argsdict['nltk_ner'], df_nltk_ner)
    # get spacy noun chunks
    df_spacy_nc = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(text_spacy_nc)(sents.sentences[i], sents.docid[i], sents.sent_id[i]) for i in
        tqdm(range(0, len(sents))))
    df_spacy_nc = [item for sublist in df_spacy_nc for item in sublist]
    df_spacy_nc_2 = pd.DataFrame.from_dict(df_spacy_nc)
    mongowrite(argsdict, argsdict['spacy_nc'], df_spacy_nc_2)
    ncdf_ver_doc_sent = df_spacy_nc_2.groupby(['docid', 'sent_id', 'Verb']).size().reset_index(name='counts')
    mongowrite(argsdict, argsdict['verbbydocsent'], ncdf_ver_doc_sent)
    ncdf_ver = df_spacy_nc_2.groupby(['Verb']).size().reset_index(name='counts')
    mongowrite(argsdict, argsdict['verb'], ncdf_ver)
    nerstan_mer, nernltk_mer, nerspacy_mer, nerpoly_mer = df_create_for_merge(stan_ner_df_final, df_nltk_ner,
                                                                              df_spacy_ner, df_poly_ner)
    if all([len(nerstan_mer) > 0, len(nerspacy_mer) > 0]):
        ner_common_1 = pd.merge(nerstan_mer, nerspacy_mer, how='outer', on=['ent', 'docid', 'sent_id'])
    elif any([len(nerstan_mer) > 0, len(nerspacy_mer) > 0]):
        ner_common_1 = nerstan_mer if len(nerspacy_mer) == 0 else nerspacy_mer
    else:
        ner_common_1 = pd.DataFrame()
    if all([len(nerpoly_mer) > 0, len(nernltk_mer) > 0]):
        ner_common_2 = pd.merge(nerpoly_mer, nernltk_mer, how='outer', on=['ent', 'docid', 'sent_id'])
    elif any([len(nerpoly_mer) > 0, len(nernltk_mer) > 0]):
        ner_common_2 = nerpoly_mer if len(nernltk_mer) == 0 else nernltk_mer
    else:
        ner_common_2 = pd.DataFrame()
    if all([len(ner_common_1) > 0, len(ner_common_2) > 0]):
        ner_common_final_df = pd.merge(ner_common_1, ner_common_2, how='outer', on=['ent', 'docid', 'sent_id'])
    elif any([len(ner_common_1) > 0, len(ner_common_2) > 0]):
        ner_common_final_df = ner_common_1 if len(ner_common_2) == 0 else ner_common_2
    else:
        ner_common_final_df = pd.DataFrame()
    ner_common_final_df = ner_common_final_df.drop_duplicates()
    mongowrite(argsdict, argsdict['entgold'], ner_common_final_df)
    #get curated sets
    # choices = list(set(list(ner_common_final_df.ent)))
    # list_ent_final = []
    # for j in tqdm(range(0, len(choices))):
    #     repltext = {}
    #     query = choices[j]
    #     # Get a list of matches ordered by score, default limit to 5
    #     repltext['match'] = list(
    #         map(lambda x: {"name": x[0], "F1": x[1]} if x[1] > 80 else "", process.extract(query, choices)))
    #     repltext['listent'] = list(set(list(
    #         map(lambda x: x[0] if x[1] >= 90 else None, process.extract(query, choices)))))
    #     repltext['query'] = query
    #     list_ent_final.append(repltext)
    # maplist = list(map(lambda x: list(set(list(map(lambda x: x.strip() if x is not None else None, x['listent'])))),
    #                    list_ent_final))
    # listcomp = []
    # for listm in tqdm(maplist):
    #     matchge = list(map(lambda x: float(len(set(listm) & set(x))) / len(listm) * 100, maplist))
    #     matchidx = [i for i in range(len(matchge)) if matchge[i] > 50]
    #     matchlol = [maplist[i] for i in matchidx]
    #     flatten = list(set([item for sublist in matchlol for item in sublist]))
    #     listcomp.append(flatten)
    # result = [list(j) for j in set(tuple(j) for j in listcomp)]
    # result = list(map(lambda x: list(filter(None, x)), result))
    # if (len(result) > 0):
    #     mongowrite(argsdict, argsdict['entcurated'], result)
    # topic modeling
    data = rawdata.cleaned_coref_text.values.tolist()
    data_words = list(sent_to_words(data))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=20, random_state=100,
                                                update_every=1, chunksize=100, passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic['Document_No'] = rawdata['doc_id']
    df_dominant_topic.dropna(inplace=True)
    tops = lda_model.print_topics()
    mongowrite(argsdict, argsdict['topic_coll'], df_dominant_topic)
    Topics = list(map(lambda x: list(
        map(lambda x: {x.split("*")[1].replace("\"", "").strip(): float(x.split("*")[0])}, x[1].split("+"))), tops))
    tops = []
    for i in range(0, len(Topics)):
        tempdict = {}
        tempdict['topicnum'] = i
        tempdict['topickeys'] = Topics[i]
        tempdict['doc_id'] = list(rawdata.doc_id)
        tops.append(tempdict)
    mongowrite(argsdict, argsdict['topics'], tops)
    # cluster modelling
    hasher = TfidfVectorizer(stop_words='english')
    vector = make_pipeline(hasher, TfidfTransformer())
    tfidf = vector.fit_transform(data)
    scr = -1
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(tfidf)
        pred = kmeans.predict(tfidf)
        centers = kmeans.cluster_centers_
        score = silhouette_score(tfidf, pred, metric='euclidean')
        if score > scr:
            k = n_clusters
    # Using KMean to cluster
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(tfidf)
    pred = kmeans.predict(tfidf)
    centers = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = hasher.get_feature_names()
    word_list = []
    for i in range(0, k):
        tempdict = {}
        tempdict[str(i)] = []
        tempdict['doc_id'] = list(rawdata.doc_id)
        for j in centers[i, :20]:
            tempdict[str(i)].append(terms[j])
        word_list.append(tempdict)
    df = pd.DataFrame()
    df['Cluster_id'] = pred
    df['document_id'] = rawdata['doc_id']
    df['text'] = rawdata['text']
    cluster_id_doc = df.drop_duplicates()
    mongowrite(argsdict, argsdict['cluster_df'], cluster_id_doc)
    mongowrite(argsdict, argsdict['cluster_word_df'], word_list)
    # create word2vec model
    wrdembedsent = list(map(lambda x: tok_text(x).split(),list(rawdata['cleaned_coref_text'])))
    model = Word2Vec(wrdembedsent, min_count=1)
    model.wv.save_word2vec_format('model.txt', binary=False)
    pass


if __name__ == "__main__":
    with open("conf/conf.json",'r') as json_file:
        argsdict = json.load(json_file)
    json_file.close()
    main(argsdict)
