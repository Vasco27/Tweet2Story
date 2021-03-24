# Essentials
import numpy as np

# Custom modules
from T2S.src.utils.data_utils import trim_whitespaces, multiple_index_list
from T2S.src.utils.json_utils import NoIndent

# coref
from allennlp.predictors.predictor import Predictor

# SPACY
import en_core_web_trf
nlp = en_core_web_trf.load()

# co-reference resolution
model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
predictor_cr = Predictor.from_path(model_url)


def coref_with_lemma(text):
    doc_topic = nlp(text)
    t_lemma = [token.lemma_ for token in doc_topic]
    topic_coref_text = trim_whitespaces(t_lemma)

    prediction = predictor_cr.predict(document=topic_coref_text)
    topic_coref_text = predictor_cr.coref_resolved(topic_coref_text)

    clusters_nouns_list = find_cluster_nouns(prediction)

    return topic_coref_text, clusters_nouns_list


def find_cluster_nouns(coref_prediction):
    cluster_noun_list = []
    for cluster in coref_prediction["clusters"]:
        cluster_list = []
        for span in cluster:
            start_idx = span[0]
            end_idx = span[1]
            indexes = np.arange(start_idx, end_idx+1).tolist()

            word_span = multiple_index_list(coref_prediction["document"], indexes)
            cluster_list.append(' '.join(word_span))
        cluster_noun_list.append(NoIndent(cluster_list))

    return cluster_noun_list
