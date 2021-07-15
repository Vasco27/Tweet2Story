# Essentials
import pandas as pd
import re
import numpy as np
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# nlp utils
import nltk
# from nltk.tokenize import sent_tokenize

# Baselines
# Utils
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
# LexRank
from lexrank import STOPWORDS, LexRank
from sumy.summarizers.lex_rank import LexRankSummarizer
# LSA
from sumy.summarizers.lsa import LsaSummarizer
# TextRank
from sumy.summarizers.text_rank import TextRankSummarizer
# Pre-trained models
# Changed torch to 1.5.1 because of AllenNLP 1.0.0
# Might cause conflicts since sentence-transformer uses torch>=1.6.0
# T5
from transformers import T5Tokenizer, T5ForConditionalGeneration
# BART
from transformers import BartTokenizer, BartForConditionalGeneration

# Evaluation
from rouge_score import rouge_scorer

nltk.download("punkt")

BASELINE_TYPES = ["multi_doc", "single_doc"]
BASELINES_SINGLE_DOC = ["LexRank", "LSA", "TextRank", "T5", "BART"]
BASELINES_MULTI_DOC = ["LexRank"]
SCORES = ["precision", "recall", "F1"]


class SummarizerHelper:
    def __init__(self, rouge_metrics=None, lang="english"):
        if rouge_metrics is None:
            rouge_metrics = ['rouge1', 'rougeL', 'rougeLsum']
            warnings.warn(f"Rouge metrics not defined, using default metrics {rouge_metrics}.")

        self.LANGUAGE = lang
        stemmer = Stemmer(self.LANGUAGE)

        # single-doc LexRank
        self.lr_sum = LexRankSummarizer(stemmer)
        self.lr_sum.stop_words = get_stop_words(self.LANGUAGE)

        # single-doc LSA
        self.lsa_sum = LsaSummarizer(stemmer)
        self.lsa_sum.stop_words = get_stop_words(self.LANGUAGE)

        # single-doc TextRank
        self.tr_sum = TextRankSummarizer(stemmer)
        self.tr_sum.stop_words = get_stop_words(self.LANGUAGE)

        # single-doc T5
        self.t5_sum_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.t5_sum_tokenizer = T5Tokenizer.from_pretrained('t5-base')

        # single-doc BART
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        # self.bart_sum_model = pipeline('summarization', model='facebook/bart-large-cnn',
        #                                tokenizer='facebook/bart-large-cnn')
        self.bart_sum_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        # SCORES
        # what is stemming? - https://en.wikipedia.org/wiki/Stemming
        # Stemming is the process of reducing words to their root form.
        # For example: contesting -> contest ; contestant -> contest
        self.scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)

    def multi_doc_sum(self, lxr, sentences, sum_size=2):
        summary = lxr.get_summary(sentences, summary_size=sum_size, threshold=.1)
        summary = '\n'.join(summary)

        return summary

    def single_doc_sum(self, summarizer, document, sum_size=2):
        parser = PlaintextParser.from_string(document, Tokenizer(self.LANGUAGE))
        summary = summarizer(parser.document, sum_size)

        str_summary = []
        for sentence in summary:
            str_summary.append(str(sentence))

        str_summary = '\n'.join(str_summary)
        return str_summary

    def pre_trained_sum(self, text, tokenizer, model, max_len_inp=512, min_len_sum=60, max_len_sum=180, len_pen=4.0):
        tokens_input = tokenizer.encode(text, return_tensors="pt", max_length=max_len_inp, truncation=True)
        summary_ids = model.generate(tokens_input, min_length=min_len_sum, max_length=max_len_sum,
                                     length_penalty=len_pen)
        summary = self.t5_sum_tokenizer.decode(summary_ids[0])
        return summary


def _make_results_df(baselines, metrics, means):
    if len(baselines) != len(list(means.values())[0]):
        raise ValueError(
            f"Length mismatch between baselines and mean scores. {len(baselines)} vs. {len(list(means.values())[0])}")

    return pd.DataFrame(means, index=baselines, columns=metrics)


def calculate_mean_scores(results, metrics, base_type="single_doc", score="F1", decimal_fields=3):
    if score not in SCORES:
        raise ValueError(f"score parameter must be one of {SCORES}. Instead it was {score}.")
    if base_type not in BASELINE_TYPES:
        raise ValueError(f"base_type parameter must be one of {BASELINE_TYPES}. Instead it was {base_type}.")

    if score == "precision":
        ind = 0
    elif score == "recall":
        ind = 1
    else:
        ind = 2

    if base_type == "single_doc":
        base_type = "single_document"
        baselines = BASELINES_SINGLE_DOC
    else:
        base_type = "multi_document"
        baselines = BASELINES_MULTI_DOC
    means = {key: [] for key in metrics}

    for baseline in baselines:
        for metric in metrics:
            mean = np.array([val[base_type]["metrics"][metric][ind] for val in results[baseline]]).mean()
            means[metric].append(round(mean, decimal_fields))

    return _make_results_df(baselines, metrics, means)
