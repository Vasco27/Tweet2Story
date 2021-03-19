# Essentials
import pandas as pd
import json
import numpy as np
import time

# Turning off the GPU (use CPU instead)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Visualization
from pprint import pprint

# Custom modules
from T2S.src.utils.data_utils import get_paths, flatten_list, multiple_index_list, trim_whitespaces
from T2S.src.utils.json_utils import NoIndent, MyEncoder

# SRL
# _jsonnet was really slowing down the process.
# pip install jsonnetbin was perfect and cleared the error.
# Performance is now at normal speed
from allennlp.predictors.predictor import Predictor
# Split text in sentences
from nltk import tokenize
# co-reference resolution
# import neuralcoref  # neuralcoref not compatible with spacy > 3.0.0

# SPACY
import en_core_web_trf
nlp = en_core_web_trf.load()

print("\nAllenNLP loading predictors...")
start = time.time()
# SRL
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)
# co-reference resolution
model_url = 'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz'
predictor_cr = Predictor.from_path(model_url)  # load the model
end = time.time()
print(f"Computation time - {round(end - start, 2)} seconds")

DECIMAL_FIGURES = 3


def verb_sem_eval(hyp_lemma, ref_lemma):
    COR = sum(verb in ref_lemma for verb in hyp_lemma)
    SPU = sum(verb not in ref_lemma for verb in hyp_lemma)
    MIS = sum(verb not in hyp_lemma for verb in ref_lemma)

    POS = COR + MIS
    ACT = COR + SPU

    precision = round(COR / ACT * 100, DECIMAL_FIGURES)
    recall = round(COR / POS * 100, DECIMAL_FIGURES)

    return precision, recall


def make_verbs_lemma(verbs_nested_lists):
    verbs_nested_lists = flatten_list(verbs_nested_lists)
    verbs_doc = nlp(' '.join(verbs_nested_lists))
    verbs_lemma = [token.lemma_ for token in verbs_doc]

    return verbs_lemma


if __name__ == '__main__':
    print("\nProgram start...")
    path_to_data, results_dir = get_paths()
    results_dir = results_dir + "baselines/SRL/"

    tweetir_data = pd.read_csv(path_to_data)
    topic_ids = tweetir_data["topic"].unique()

    results_list = []

    topic = "5811057c-6732-4b37-b04c-ddf0a75a7b51"

    topic_data = tweetir_data[tweetir_data["topic"] == topic]

    # Data on specific topic
    topic_content = topic_data["topics.content"].unique()[0]
    topic_sentences = tokenize.sent_tokenize(topic_content)

    tweet_multi_doc = topic_data["tweets.full_text"].apply(str.strip).tolist()
    tweets_single_doc = '.\n'.join(tweet_multi_doc)
    tweet_sentences = tokenize.sent_tokenize(tweets_single_doc)

    data_row = {
        "topic": topic, "content": topic_content, "coref_content": None,
        "tweets": tweet_multi_doc, "coref_tweets": None, "nr_tweets": len(tweet_multi_doc),
        "topic_clusters": [], "tweets_clusters": [],
        "topic_sentences_srl": [], "tweets_sentences_srl": [],
        "metrics": {
            "verb_precision": -1, "verb_recall": -1
        }
    }

    ###########################
    # Co-Reference Resolution #
    ###########################

    print("\nCo-Reference Resolution...")
    start = time.time()
    # Topic
    # todo: script function to abstract from tweets and topics should return the coref_text and the cluster_list
    #  to assign it to the data_row inside the main runtime
    doc_topic = nlp(topic_content)
    t_lemma = [token.lemma_ for token in doc_topic]
    topic_coref_text = trim_whitespaces(t_lemma)

    prediction = predictor_cr.predict(document=topic_coref_text)
    topic_coref_text = predictor_cr.coref_resolved(topic_coref_text)
    data_row["coref_content"] = topic_coref_text

    cluster_list = []
    for span in prediction["clusters"][0]:
        start_idx = span[0]
        end_idx = span[1]
        indexes = np.arange(start_idx, end_idx+1).tolist()

        word_span = multiple_index_list(prediction["document"], indexes)
        cluster_list.append(' '.join(word_span))
    data_row["topic_clusters"] = NoIndent(cluster_list)

    # Tweets
    doc_tweets = nlp(tweets_single_doc)
    tw_lemma = [token.lemma_ for token in doc_tweets]
    tweets_coref_text = trim_whitespaces(tw_lemma)

    prediction = predictor_cr.predict(document=tweets_coref_text)
    coref_tweets = predictor_cr.coref_resolved(tweets_coref_text)
    data_row["coref_tweets"] = coref_tweets

    cluster_list = []
    for span in prediction["clusters"][0]:
        start_idx = span[0]
        end_idx = span[1]
        indexes = np.arange(start_idx, end_idx + 1).tolist()

        word_span = multiple_index_list(prediction["document"], indexes)
        cluster_list.append(' '.join(word_span))
    data_row["tweets_clusters"] = NoIndent(cluster_list)

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")

    #######
    # SRL #
    #######

    print("\nSemantic Role Labelling...")
    start = time.time()
    # Topic
    topic_verbs = []
    for topic_sentence in topic_sentences:
        result = predictor.predict_json(
            {"sentence": topic_sentence}
        )

        topic_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        topic_verbs.append(verbs)
        # pprint(topic_sent, indent=2)
        data_row["topic_sentences_srl"].append(topic_sent)

    # Topic lemmatization
    topic_lemma = make_verbs_lemma(topic_verbs)

    # Tweets
    tweets_verbs = []
    for tweet_sentence in tweet_sentences:
        result = predictor.predict_json(
            {"sentence": tweet_sentence}
        )

        tweet_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        tweets_verbs.append(verbs)
        # pprint(tweet_sent, indent=2)
        data_row["tweets_sentences_srl"].append(tweet_sent)

    # Tweets lemmatization
    tweets_lemma = make_verbs_lemma(tweets_verbs)

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")

    ##############
    # Evaluation #
    ##############

    print("\nEvaluation...")
    # Verb evaluation
    verb_precision, verb_recall = verb_sem_eval(tweets_lemma, topic_lemma)
    data_row["metrics"]["verb_precision"] = verb_precision
    data_row["metrics"]["verb_recall"] = verb_recall

    print(f"\nTopic vs Tweets verb recall is {verb_recall} %")

    results_list.append(data_row)

    print("\nExporting to JSON...")
    with open(results_dir + "srl_exp1.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))
