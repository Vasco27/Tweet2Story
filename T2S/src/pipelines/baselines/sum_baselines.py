# Essentials
import pandas as pd
import re
import json
import time

# nlp utils
import nltk
# from nltk.tokenize import sent_tokenize

# Custom utils
from T2S.src.utils.data_utils import get_paths
import T2S.src.utils.sum_base_utils as utils
from T2S.src.utils.sum_base_utils import SummarizerHelper

nltk.download("punkt")


def sum_base_pipeline(topic, tweets, topics, baseline="LexRank", base_type="multi_doc", summarizer=None, tokenizer=None,
                      sum_size=(2, 3), max_len_inp=(300, 512), min_len_sum=(60, 100), max_len_sum=(180, 300),
                      len_pen=(4.0, 6.0)):
    if base_type == "multi_doc":
        if baseline == "LexRank":
            tweet_summary = SUMMARIZERS.multi_doc_sum(SUMMARIZERS.multi_lr_tweet_sum, tweets, sum_size=sum_size[0])
            topic_summary = SUMMARIZERS.multi_doc_sum(SUMMARIZERS.multi_lr_topic_sum, topics, sum_size=sum_size[1])
        else:
            raise ValueError(f"baseline for type {base_type} must be one of {utils.BASELINES_MULTI_DOC}")
    elif base_type == "single_doc":
        if baseline in ["LexRank", "LSA", "TextRank"]:
            tweet_summary = SUMMARIZERS.single_doc_sum(summarizer, tweets, sum_size=sum_size[0])
            topic_summary = SUMMARIZERS.single_doc_sum(summarizer, topics, sum_size=sum_size[1])
        elif baseline in ["T5", "BART"]:
            tweet_summary = SUMMARIZERS.pre_trained_sum(tweets, tokenizer, summarizer, max_len_inp=max_len_inp[0],
                                                        min_len_sum=min_len_sum[0], max_len_sum=max_len_sum[0],
                                                        len_pen=len_pen[0])
            topic_summary = SUMMARIZERS.pre_trained_sum(topics, tokenizer, summarizer, max_len_inp=max_len_inp[1],
                                                        min_len_sum=min_len_sum[1], max_len_sum=max_len_sum[1],
                                                        len_pen=len_pen[1])
        else:
            raise ValueError(f"baseline for type {base_type} must be one of {utils.BASELINES_SINGLE_DOC}")
    else:
        raise ValueError(f"baseline type must be one of {utils.BASELINE_TYPES}. Instead it was {base_type}.")

    scores = SUMMARIZERS.scorer.score(tweet_summary, topic_summary)
    return {"topic": topic, "topic_summary": topic_summary, "tweets_summary": tweet_summary, "metrics": scores}


if __name__ == '__main__':
    # working directory
    path_to_data, results_dir = get_paths()
    results_dir = results_dir + "baselines/sumarization/"

    print(f"\nReading data from {path_to_data}...\n")
    tweetir_data = pd.read_csv(path_to_data)

    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    # Define summarizers
    print("\nDefining summarizers (from SummarizerHelper)...")
    SUMMARIZERS = SummarizerHelper(tweetir_data["tweets.full_text"], tweetir_data["topics.content"].unique(),
                                   rouge_metrics=metrics, lang="english")

    # make main loop
    topics = tweetir_data["topic"].unique()
    results = {"LexRank": [], "LSA": [], "TextRank": [], "T5": [], "BART": []}
    iteration = 1
    total_iterations = topics.shape[0]
    for topic in topics:
        start = time.time()
        print(f"Iteration {iteration} of {total_iterations}")
        topic_data = tweetir_data[tweetir_data["topic"] == topic]

        # Tweets
        tweet_multi_doc = topic_data["tweets.full_text"].tolist()
        tweets_single_doc = ' '.join(tweet_multi_doc)

        # Topics
        topic_single_doc = topic_data["topics.content"].unique().tolist()[0]
        topic_multi_doc = re.split(r'[.?!][.?!\s]{2}', topic_single_doc)  # could try with nltk.sent_tokenize()

        # Multi-doc LexRank
        multi_doc = sum_base_pipeline(topic, tweet_multi_doc, topic_multi_doc, baseline="LexRank",
                                      base_type="multi_doc", sum_size=(2, 3))

        # Single-doc LexRank
        single_doc = sum_base_pipeline(topic, tweets_single_doc, topic_single_doc, baseline="LexRank",
                                       base_type="single_doc", summarizer=SUMMARIZERS.lr_sum, sum_size=(2, 3))

        results["LexRank"].append({"single_document": single_doc, "multi_document": multi_doc})

        # Single-doc LSA
        single_doc = sum_base_pipeline(topic, tweets_single_doc, topic_single_doc, baseline="LSA",
                                       base_type="single_doc", summarizer=SUMMARIZERS.lsa_sum, sum_size=(2, 3))
        results["LSA"].append({"single_document": single_doc})

        # Single-doc TextRank
        single_doc = sum_base_pipeline(topic, tweets_single_doc, topic_single_doc, baseline="TextRank",
                                       base_type="single_doc", summarizer=SUMMARIZERS.tr_sum, sum_size=(2, 3))
        results["TextRank"].append({"single_document": single_doc})

        # Single-doc T5
        single_doc = sum_base_pipeline(topic, tweets_single_doc, topic_single_doc, baseline="T5",
                                       base_type="single_doc", summarizer=SUMMARIZERS.t5_sum_model,
                                       tokenizer=SUMMARIZERS.t5_sum_tokenizer, max_len_inp=(300, 512),
                                       min_len_sum=(60, 100), max_len_sum=(180, 300), len_pen=(4.0, 6.0))
        results["T5"].append({"single_document": single_doc})

        # Single-doc BART
        single_doc = sum_base_pipeline(topic, tweets_single_doc, topic_single_doc, baseline="BART",
                                       base_type="single_doc", summarizer=SUMMARIZERS.bart_sum_model,
                                       tokenizer=SUMMARIZERS.bart_tokenizer, max_len_inp=(300, 512),
                                       min_len_sum=(60, 100), max_len_sum=(180, 300), len_pen=(4.0, 6.0))
        results["BART"].append({"single_document": single_doc})

        iteration += 1
        end = time.time()
        print(f"Iteration time - {round(end - start, 2)} seconds.")

    # Dump json with all summaries and metrics for every topic
    print(f"\nDumping results to json file {results_dir + 'sum_baselines_ext.json'}...")
    with open(results_dir + "sum_baselines_ext.json", "w") as f:
        json.dump(results, f, indent=4)

    # Calculate mean resultfiles
    print(f"\nCalculating mean scores and exporting to file {results_dir + 'mean_sum_baselines_fscores.csv'}...")
    mean_f1_df = utils.calculate_mean_scores(results, metrics, base_type="single_doc", score="F1", decimal_fields=3)
    mean_f1_df.to_csv(results_dir + "mean_sum_baselines_fscores.csv")
