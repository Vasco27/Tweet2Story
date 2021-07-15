# Essentials
import pandas as pd
import re
import json
import time
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# nlp utils
import nltk
# from nltk.tokenize import sent_tokenize

# Custom utils
from T2S.src.utils.data_utils import get_paths
import T2S.src.utils.sum_base_utils as utils
from T2S.src.utils.sum_base_utils import SummarizerHelper

nltk.download("punkt")

ROOT_DIR = os.path.join(Path(__file__).parent.parent.parent.parent)
RESULTS_DIR = os.path.join(ROOT_DIR, "resultfiles", "baselines", "summarization")
TWEETS_DIR = os.path.join(ROOT_DIR, "resultfiles", "tweets_final")
NEWS_DIR = os.path.join(ROOT_DIR, "resultfiles", "news_final")


def sum_base_pipeline(topic, tweets, topics, baseline="LexRank", base_type="multi_doc", summarizer=None, tokenizer=None,
                      sum_size=(2, 3), max_len_inp=(300, 512), min_len_sum=(60, 100), max_len_sum=(180, 300),
                      len_pen=(4.0, 6.0)):
    if base_type == "single_doc":
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

    scores = SUMMARIZERS.scorer.score(topic_summary, tweet_summary)
    return {"topic": topic, "topic_summary": topic_summary, "tweets_summary": tweet_summary, "metrics": scores}


if __name__ == '__main__':
    # Define summarizers
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    print("\nDefining summarizers (from SummarizerHelper)...")
    SUMMARIZERS = SummarizerHelper(rouge_metrics=metrics, lang="english")

    # make main loop
    results = {"LexRank": [], "LSA": [], "TextRank": [], "T5": [], "BART": []}
    iteration = 1
    total_iterations = len(os.listdir(NEWS_DIR))
    for filename in os.listdir(NEWS_DIR):
        start = time.time()
        topic_id = filename.split(".")[0]
        with open(os.path.join(NEWS_DIR, filename), "r", encoding="utf-8") as f:
            news = f.readlines()
        with open(os.path.join(TWEETS_DIR, filename), "r", encoding="utf-8") as f:
            tweets = f.readlines()

        news = ''.join(news)
        tweets = ''.join(tweets)
        print(f"Iteration {iteration} of {total_iterations}")

        # Single-doc LexRank
        single_doc = sum_base_pipeline(topic_id, tweets, news, baseline="LexRank",
                                       base_type="single_doc", summarizer=SUMMARIZERS.lr_sum, sum_size=(2, 3))
        results["LexRank"].append({"single_document": single_doc})

        # Single-doc LSA
        single_doc = sum_base_pipeline(topic_id, tweets, news, baseline="LSA",
                                       base_type="single_doc", summarizer=SUMMARIZERS.lsa_sum, sum_size=(2, 3))
        results["LSA"].append({"single_document": single_doc})

        # Single-doc TextRank
        single_doc = sum_base_pipeline(topic_id, tweets, news, baseline="TextRank",
                                       base_type="single_doc", summarizer=SUMMARIZERS.tr_sum, sum_size=(2, 3))
        results["TextRank"].append({"single_document": single_doc})

        # Single-doc T5
        single_doc = sum_base_pipeline(topic_id, tweets, news, baseline="T5",
                                       base_type="single_doc", summarizer=SUMMARIZERS.t5_sum_model,
                                       tokenizer=SUMMARIZERS.t5_sum_tokenizer, max_len_inp=(300, 512),
                                       min_len_sum=(60, 100), max_len_sum=(180, 300), len_pen=(4.0, 6.0))
        results["T5"].append({"single_document": single_doc})

        # Single-doc BART
        single_doc = sum_base_pipeline(topic_id, tweets, news, baseline="BART",
                                       base_type="single_doc", summarizer=SUMMARIZERS.bart_sum_model,
                                       tokenizer=SUMMARIZERS.bart_tokenizer, max_len_inp=(300, 512),
                                       min_len_sum=(60, 100), max_len_sum=(180, 300), len_pen=(4.0, 6.0))
        results["BART"].append({"single_document": single_doc})

        iteration += 1
        end = time.time()
        print(f"Iteration time - {round(end - start, 2)} seconds.")

    # Dump json with all summaries and metrics for every topic
    # print(f"\nDumping results to json file {results_dir + 'sum_baselines_ext.json'}...")
    # with open(results_dir + "sum_baselines_ext.json", "w") as f:
    #     json.dump(results, f, indent=4)

    # Calculate mean resultfiles
    print(f"\nCalculating mean fscores and exporting to file {os.path.join(RESULTS_DIR, 'mean_sum_baselines_fscores.csv')}...")
    mean_f1_df = utils.calculate_mean_scores(results, metrics, base_type="single_doc", score="F1", decimal_fields=3)
    mean_f1_df.to_csv(os.path.join(RESULTS_DIR, 'mean_sum_baselines_fscores.csv'))

    print(f"\nCalculating mean recall and exporting to file {os.path.join(RESULTS_DIR, 'mean_sum_baselines_recall.csv')}...")
    mean_recall_df = utils.calculate_mean_scores(results, metrics, base_type="single_doc", score="recall", decimal_fields=3)
    mean_recall_df.to_csv(os.path.join(RESULTS_DIR, 'mean_sum_baselines_recall.csv'))
