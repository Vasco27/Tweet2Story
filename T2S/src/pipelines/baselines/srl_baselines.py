# Essentials
import pandas as pd
import json

# Visualization
from pprint import pprint

# Custom modules
from T2S.src.utils.data_utils import get_paths

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
        "topic_sentences_srl": [], "tweets_sentences_srl": []  # aplicar a varias sentences
    }

    doc = nlp(topic_content)

    sentence_nr = 0
    topic_verbs = []
    for topic_sentence in topic_sentences:
        result = predictor.predict_json(
            {"sentence": topic_sentence}
        )

        # pprint(result, indent=2)
        topic_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        topic_verbs.append(verbs)
        # pprint(topic_sent, indent=2)
        data_row["topic_sentences_srl"].append(topic_sent)
        sentence_nr += 1
    # Topic lemmatization
    topic_verbs = [item for sublist in topic_verbs for item in sublist]
    topic_verbs_doc = nlp(' '.join(topic_verbs))
    topic_lemma = [token.lemma_ for token in topic_verbs_doc]

    sentence_nr = 0
    tweets_verbs = []
    for tweet_sentence in tweet_sentences:
        result = predictor.predict_json(
            {"sentence": tweet_sentence}
        )

        # pprint(result, indent=2)
        tweet_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        tweets_verbs.append(verbs)
        # pprint(tweet_sent, indent=2)
        data_row["tweets_sentences_srl"].append(tweet_sent)
    # Tweets lemmatization
    tweets_verbs = [item for sublist in tweets_verbs for item in sublist]
    tweets_verbs_doc = nlp(' '.join(tweets_verbs))
    tweets_lemma = [token.lemma_ for token in tweets_verbs_doc]

    # Verb recall:
    # SPU -> verb in tweets but not in topic
    # MIS -> verb in topic but not in tweets
    # COR -> verb in tweets and in topic
    # Make Exact recall:
    # POS = COR + MIS
    # ACT = COR + SPU
    # Precision = COR / ACT
    # Recall = COR / POS

    results_list.append(data_row)

    with open(results_dir + "srl_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, ensure_ascii=False, indent=4))
