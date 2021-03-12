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
        "topic_sentences_srl": {"sentence1": []}, "tweets_sentences_srl": {"sentence1": []}  # aplicar a varias sentences
    }

    result = predictor.predict_json(
        {"sentence": topic_sentences[0]}
    )

    # pprint(result, indent=2)
    topic_sent = [frame["description"] for frame in result["verbs"]]
    pprint(topic_sent, indent=2)
    data_row["topic_sentences_srl"]["sentence1"] = topic_sent

    for tweet_sentence in tweet_sentences:
        result = predictor.predict_json(
            {"sentence": tweet_sentence}
        )

        # pprint(result, indent=2)
        tweet_sent = [frame["description"] for frame in result["verbs"]]
        # pprint(tweet_sent, indent=2)
        data_row["tweets_sentences_srl"]["sentence1"].append(tweet_sent)

    results_list.append(data_row)

    with open(results_dir + "srl_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, ensure_ascii=False, indent=4))
