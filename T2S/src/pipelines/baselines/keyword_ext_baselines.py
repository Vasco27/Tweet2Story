# Essentials
import pandas as pd
import os
import json
import time
import warnings
from collections import namedtuple
from pathlib import Path

# Keyword extraction methods
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
from yake import KeywordExtractor
from summa.keywords import keywords as TextRank_kw
from pke.unsupervised import TopicRank
from keybert import KeyBERT

# Visualization
# import re
# from pprint import pprint

# Custom utils
from T2S.src.utils.eval_utils import compute_jaccard_index, keyphrase_eval_metrics
from T2S.src.utils.json_utils import NoIndent, MyEncoder

nltk.download("stopwords")
nltk.download("universal_tagset")
spacy.load("en_core_web_sm")

BASELINES = ["TFIDF", "RAKE", "YAKE", "TextRank", "TopicRank", "KeyBert"]
ROUGE_METRICS = ["rouge1"]
METRICS = ["precision", "recall", "jaccard", "rouge1_precision", "rouge1_fscore", "R_precision"]

DECIMAL_FIGURES = 3
TOP_N_KEYWORDS = 10
MAX_KEYWORDS = 3
MIN_KEYWORDS = 1

ROOT_DIR = os.path.join(Path(__file__).parent.parent.parent.parent)
RESULTS_DIR = os.path.join(ROOT_DIR, "resultfiles", "baselines", "keyword_extraction")
TWEETS_DIR = os.path.join(ROOT_DIR, "resultfiles", "tweets_final")
NEWS_DIR = os.path.join(ROOT_DIR, "resultfiles", "news_final")
TEMP_DIR = os.path.join(ROOT_DIR, "datafiles", "temp")


def make_tfidf_matrix(docs, topic_ids, max_df=0.95, min_df=0.00):
    vectorizer = TfidfVectorizer(stop_words="english", max_df=max_df, min_df=min_df)

    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    dense_list = dense.tolist()
    tfidf_df = pd.DataFrame(dense_list, columns=feature_names, index=topic_ids)

    return tfidf_df


def topic_rank_kw_extraction(temp_file, text):
    # Save text to temporary '.txt' file (it's the only way it works with the pke library)
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)

    tr = TopicRank()
    tr.load_document(temp_file)
    tr.candidate_selection()
    tr.candidate_weighting()
    keywords = tr.get_n_best(n=TOP_N_KEYWORDS)
    keys = [kw for kw, _ in keywords]

    return keys


def evaluate_baseline(ref_kw_freq, hyp_kw_freq, fill_results_dict=False, base_name=None):
    if base_name not in BASELINES:
        raise ValueError(f"Baseline {base_name} must be one of the implemented baselines:\n{BASELINES}")
    if fill_results_dict & (base_name is None):
        raise ValueError("If parameter 'fill_results_dict' is set, then parameter 'base_name' must be given.")

    # Reference - topic | hypothesis - tweets
    ref_kw = [value.lower() for d in ref_kw_freq for (key, value) in d.items() if key == "keyword"]
    hyp_kw = [value.lower() for d in hyp_kw_freq for (key, value) in d.items() if key == "keyword"]

    # Precision - how many similar keywords?
    precision = sum(t_key in hyp_kw for t_key in ref_kw) / len(ref_kw)
    precision = round(precision * 100, DECIMAL_FIGURES)

    # recall - how many kw from the news were actual retrieved by the tweets
    recall = sum(h_key in ref_kw for h_key in hyp_kw) / len(ref_kw)
    recall = round(recall * 100, DECIMAL_FIGURES)

    # Jaccard index between two sets of keywords
    jaccard = compute_jaccard_index(set(ref_kw), set(hyp_kw), DECIMAL_FIGURES)

    # data_row contains the resultfiles of each baseline
    if fill_results_dict:
        data_row["baselines"][base_name]["topic_keywords"]["kw_list"] = NoIndent(ref_kw)
        data_row["baselines"][base_name]["tweets_keywords"]["kw_list"] = NoIndent(hyp_kw_freq)
        data_row["baselines"][base_name]["metrics"]["precision"] = precision
        data_row["baselines"][base_name]["metrics"]["recall"] = recall
        data_row["baselines"][base_name]["metrics"]["jaccard"] = jaccard

        keyphrase_eval_metrics(ref_kw, hyp_kw, data_row, fill_results_dict=fill_results_dict, base_name=base_name,
                               decimal_figures=DECIMAL_FIGURES, rouge_metrics=ROUGE_METRICS)
    else:
        return precision, jaccard

    return


def nr_keywords_in_text(keywords, text, ref_text):
    """
    Counts the number of times a keyword appears in a text.
    Also calculates the importance of the word (or words) in the text, through a weighted recall.

    Weighted Recall: Weighted relative frequency to calculate the importance of a keyword in both texts
    Numerator - (WordInText(W1) + WordInRefText) * freq(9+5) /
    Denominator - (WordsInText(W1+W2+W3) + WordsInRefText(W1+W2+W3)) * freq((9+5+4)+(2+5+1))

    :param keywords: list of keywords extracted from a document
    :param text: the document from where to calculate the importance of the word through the weighted recall
    :param ref_text: the document from where the keywords are extracted
    :return: list of (keyword, frequency, weighted_recall) tuples

    Note: The named tuple 'Keyword' is converted to 'dict' object, in order to be serializable to JSON.
    We use named tuple instead of a normal dict to make the Keyword a less arbitrary concept.
    """
    kw_freq_list = []
    low_text = text.lower()
    low_ref_text = ref_text.lower()

    text_freqs = [low_text.count(key) for key in keywords]
    total_kw_freq = sum(text_freqs)

    ref_text_freqs = [low_ref_text.count(key) for key in keywords]
    total_kw_ref_freq = sum(ref_text_freqs)

    for kw, freq, ref_freq in zip(keywords, text_freqs, ref_text_freqs):
        if total_kw_freq != 0:
            weighted_recall = round((freq + ref_freq) / (total_kw_freq + total_kw_ref_freq), DECIMAL_FIGURES)
        else:
            weighted_recall = 0

        kw_freq_list.append(Keyword(keyword=kw, freq=freq, w_recall=weighted_recall)._asdict())

    return kw_freq_list


if __name__ == '__main__':
    tweet_docs, topic_docs, topic_ids = [], [], []
    for filename in os.listdir(NEWS_DIR):
        topic_id = filename.split(".")[0]
        topic_ids.append(topic_id)
        with open(os.path.join(NEWS_DIR, filename), "r", encoding="utf-8") as f:
            news = f.readlines()
        with open(os.path.join(TWEETS_DIR, filename), "r", encoding="utf-8") as f:
            tweets = f.readlines()
        topic_docs.append(''.join(news))
        tweet_docs.append(''.join(tweets))

    topics_tfidf_matrix = make_tfidf_matrix(topic_docs, topic_ids)
    tweets_tfidf_matrix = make_tfidf_matrix(tweet_docs, topic_ids)

    # named tuple for keyword representation in json
    Keyword = namedtuple("Keyword", ["keyword", "freq", "w_recall"])

    results_list = []

    # min_length and max_length to force the keywords to be just one word
    # Try with 3 words (it is said to be the optimal length)
    r = Rake(language="english", min_length=1, max_length=1)
    # n=1 to force kw to be one word | top=10 to get top 10 best ranked kw
    # Try with n=3
    yake = KeywordExtractor(lan="en", n=1, top=TOP_N_KEYWORDS)

    key_bert = KeyBERT('distilbert-base-nli-mean-tokens')

    iteration, total_iterations = 1, len(topic_ids)
    # Begin topic cycle
    for filename in os.listdir(NEWS_DIR):
        start = time.time()
        topic_id = filename.split(".")[0]
        with open(os.path.join(NEWS_DIR, filename), "r", encoding="utf-8") as f:
            news = f.readlines()
        with open(os.path.join(TWEETS_DIR, filename), "r", encoding="utf-8") as f:
            tweets = f.readlines()

        n_tweets = len(tweets)
        news = ''.join(news)
        tweets = ''.join(tweets)

        print(f"Iteration {iteration} of {total_iterations}\nTopic - {topic_id}")

        # structure resultfiles for json dump
        data_row = {"topic": topic_id, "content": news, "nr_tweets": n_tweets, "baselines": {
            base: {"topic_keywords": {"kw_list": []}, "tweets_keywords": {"kw_list": []},
                   "metrics": {"precision": -1, "recall": -1, "jaccard": -1, "rouge1_precision": -1,
                               "rouge1_fscore": -1, "R_precision": -1}} for base in BASELINES
        }}

        # print("\nTOPIC CONTENT:")
        # pprint(re.sub("[ ]{2,}", "\n", topics_content), width=200)

        # print("\nTWEETS RELATED TO THE TOPIC:")
        # pprint(tweets, width=200)

        # Topic TF-IDF matrix and top 10 keywords
        topic_tfidf_matrix = topics_tfidf_matrix.loc[topic_id]
        topic_keywords = topic_tfidf_matrix.nlargest(TOP_N_KEYWORDS)
        t_keys = topic_keywords.index.tolist()
        t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        # Tweet TF-IDF matrix and top 10 keywords
        tweets_keywords = tweets_tfidf_matrix.loc[topic_id].nlargest(TOP_N_KEYWORDS)
        tw_keys = tweets_keywords.index.tolist()
        tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        # Evaluate baseline with given metrics and fill the resultfiles dict
        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="TFIDF")

        # RAKE
        r.extract_keywords_from_text(news)
        topic_keywords = r.get_ranked_phrases()
        t_keys = topic_keywords[:TOP_N_KEYWORDS]
        t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        r.extract_keywords_from_text(tweets)
        tweets_keywords = r.get_ranked_phrases()
        tw_keys = tweets_keywords[:TOP_N_KEYWORDS]
        tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="RAKE")

        # YAKE!
        # The lower the score, the more relevant the keywords
        topic_keywords = yake.extract_keywords(news)
        t_keys = [kw for kw, _ in topic_keywords]
        t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        tweets_keywords = yake.extract_keywords(tweets)
        tw_keys = [kw for kw, _ in tweets_keywords]
        tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="YAKE")

        # TextRank
        # words=20 to take approximately 20 keywords (not certain) | split to force kw into a list
        try:
            topic_keywords = TextRank_kw(news, words=TOP_N_KEYWORDS, language="english", split=True)
            t_keys = topic_keywords[:TOP_N_KEYWORDS]
            t_kw_freq = nr_keywords_in_text(t_keys, news, news)
        except IndexError:
            warnings.warn("Topic has less than 10 keywords")
            topic_keywords = TextRank_kw(news, words=1, language="english", split=True)
            t_keys = topic_keywords
            t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        try:
            tweets_keywords = TextRank_kw(tweets, words=TOP_N_KEYWORDS, language="english", split=True)
            tw_keys = tweets_keywords[:TOP_N_KEYWORDS]
            tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)
        except IndexError:
            warnings.warn("Tweets has less than 10 keywords")
            tweets_keywords = TextRank_kw(tweets, words=1, language="english", split=True)
            tw_keys = tweets_keywords
            tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="TextRank")

        # TopicRank
        t_keys = topic_rank_kw_extraction(os.path.join(TEMP_DIR, "news_temp.txt"), news)
        t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        tw_keys = topic_rank_kw_extraction(os.path.join(TEMP_DIR, "news_temp.txt"), tweets)
        tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="TopicRank")

        # Pre-trained KeyBert
        topic_keywords = key_bert.extract_keywords(news, top_n=TOP_N_KEYWORDS, keyphrase_ngram_range=(1, 2))
        t_keys = [kw for kw, _ in topic_keywords]
        t_kw_freq = nr_keywords_in_text(t_keys, news, news)

        tweets_keywords = key_bert.extract_keywords(tweets, top_n=TOP_N_KEYWORDS,
                                                    keyphrase_ngram_range=(1, 2))
        tw_keys = [kw for kw, score in tweets_keywords]
        tw_kw_freq = nr_keywords_in_text(tw_keys, news, tweets)

        evaluate_baseline(t_kw_freq, tw_kw_freq, fill_results_dict=True, base_name="KeyBert")

        results_list.append(data_row)

        end = time.time()
        print(f"Iteration time - {round(end - start, 2)} seconds.")
        iteration += 1

    print(f"JSON keyword extraction file exported to - {os.path.join(RESULTS_DIR, 'keywords_extraction_baselines.json')}")
    with open(os.path.join(RESULTS_DIR, 'keywords_extraction_baselines.json'), "w", encoding="utf-8") as file:
        file.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))

    # CALCULATE THE MEAN FOR EACH METRIC IN EACH BASELINE (for later visualization)
    results_mean_df = pd.DataFrame(index=BASELINES, columns=METRICS)
    topic_with_kp_list = []
    for metric in METRICS:
        metric_mean_list = []
        for baseline in BASELINES:
            metric_total = float(sum(d["baselines"][baseline]["metrics"][metric] for d in results_list if
                                     d["baselines"][baseline]["metrics"][metric] != -1)) 

            metric_mean = round(metric_total / total_iterations, DECIMAL_FIGURES)
            metric_mean_list.append(metric_mean)

        results_mean_df[metric] = metric_mean_list
    results_mean_df["total_topics"] = [total_iterations] * results_mean_df.shape[0]

    print(f"Mean results for keyword extraction evaluation metrics exported to - "
          f"{os.path.join(RESULTS_DIR, 'mean_kw_baselines.csv')}")
    results_mean_df.to_csv(os.path.join(RESULTS_DIR, 'mean_kw_baselines.csv'))
