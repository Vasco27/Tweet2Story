# Essentials
import pandas as pd
import json

# Visualization
from pprint import pprint

# Custom modules
from T2S.src.utils.data_utils import get_paths
from T2S.src.utils.eval_utils import semeval_confusion_matrix, semeval_metrics_computation
from T2S.src.utils.json_utils import MyEncoder, NoIndent

# Time Tagging methods
from py_heideltime import py_heideltime

DECIMAL_FIGURES = 3


if __name__ == '__main__':
    # tt = time tagging
    path_to_data, results_dir = get_paths()
    results_dir = results_dir + "baselines/TT/"

    tweetir_data = pd.read_csv(path_to_data)
    topic_ids = tweetir_data["topic"].unique()

    results_list = []

    topic = "0b3bea50-3a2c-4d07-953e-45aca9988634"
    """
    0b3bea50-3a2c-4d07-953e-45aca9988634 - "Good" example (some overlaps)
    be8e1b0d-6512-495d-8b55-40b0320a513e - No entities recognized. Fails to tag time expressions on tweets
    bf88166e-76fe-4605-97df-bb63f5a1806c - More text in the tweets than in the topic. 
    Still almost no time entities recognized.
    5811057c-6732-4b37-b04c-ddf0a75a7b51 - Lydia Ko golf news. Disappointing resultfiles.
    
    Conclusion: 
    - Tweets have very little time information and the ones they have are hard to extract.
    - Even when there is overlap with the topic, it's usually not significant.
    """

    topic_data = tweetir_data[tweetir_data["topic"] == topic]

    # Data on specific topic
    topic_content = topic_data["topics.content"].unique()[0]
    topic_date = topic_data["topics.published"].unique()[0]
    topic_date = pd.to_datetime(topic_date).date().isoformat()

    tweet_multi_doc = topic_data["tweets.full_text"].tolist()
    tweets_single_doc = '\n'.join(tweet_multi_doc)

    print(topic_date)

    data_row = {
        "topic": topic, "content": topic_content, "tweets": tweet_multi_doc, "nr_tweets": len(tweet_multi_doc),
        "tt_topic": [], "tt_tweets": []
    }

    tt_topic = py_heideltime(
        topic_content, date_granularity="full", document_type="news", document_creation_time=topic_date
    )
    data_row["tt_topic"] = NoIndent(tt_topic[0])
    pprint(tt_topic[0])
    # pprint(tt_topic[1], width=200)
    # pprint(tt_topic[2], width=200)
    pprint(tt_topic[3])

    tweet_tt_list = []
    for row in topic_data.itertuples():
        tweet_time = pd.to_datetime(row[4]).date().isoformat()
        tt_tweets = py_heideltime(
            row[5], date_granularity="full", document_type="colloquial", document_creation_time=tweet_time
        )
        if len(tt_tweets[0]) != 0:
            tweet_tt_list.append(tt_tweets[0])
            pprint(tt_tweets[0])
            pprint(tt_tweets[1], width=200)
            pprint(tt_tweets[3])

    tweet_tt_list = [item for sublist in tweet_tt_list for item in sublist]
    data_row["tt_tweets"] = NoIndent(tweet_tt_list)

    # Safeguard for cases where the model does not find time entities
    metrics_schema = {
        "exact_precision": 0,
        "exact_recall": 0,
        "exact_F1": 0,
        "partial_precision": 0,
        "partial_recall": 0,
        "partial_F1": 0
    }
    if len(tweet_tt_list) != 0:
        eval_schema = semeval_confusion_matrix(tt_topic[0], tweet_tt_list)
        metrics_schema = semeval_metrics_computation(eval_schema, DECIMAL_FIGURES)

    data_row = {**data_row, **metrics_schema}
    results_list.append(data_row)

    with open(results_dir + "tt_exp1.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))
