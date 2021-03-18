# Essentials
import pandas as pd
import json

# Visualization
from pprint import pprint

# Custom modules
from T2S.src.utils.data_utils import get_paths, flatten_list

# SRL
from allennlp.predictors.predictor import Predictor
# Split text in sentences
from nltk import tokenize
# Lemmatization
import en_core_web_trf
nlp = en_core_web_trf.load()

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)

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
        "topic": topic, "content": topic_content, "tweets": tweet_multi_doc, "nr_tweets": len(tweet_multi_doc),
        "topic_sentences_srl": [], "tweets_sentences_srl": [],
        "metrics": {
            "verb_precision": -1.0, "verb_recall": -1.0
        }
    }

    #######
    # SRL #
    #######

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

    ##############
    # Evaluation #
    ##############

    # Verb evaluation
    verb_precision, verb_recall = verb_sem_eval(tweets_lemma, topic_lemma)
    data_row["metrics"]["verb_precision"] = verb_precision
    data_row["metrics"]["verb_recall"] = verb_recall

    print(f"\nTopic vs Tweets verb recall is {verb_recall} %")

    results_list.append(data_row)

    with open(results_dir + "srl_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, ensure_ascii=False, indent=4))
