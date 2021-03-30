# Essentials
import pandas as pd
import json
import time

# Turning off the GPU (use CPU instead)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Visualization
from pprint import pprint

# Custom modules
from T2S.src.utils.data_utils import get_paths
from T2S.src.utils.json_utils import NoIndent, MyEncoder
from T2S.src.utils.string_utils import flatten_list, value_indexes_in_list, multiple_index_list, trim_whitespaces

# Evaluation
from rouge_score import rouge_scorer

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
from T2S.src.utils.coref_utils import coref_with_lemma
# SRL
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
)
end = time.time()
print(f"Computation time - {round(end - start, 2)} seconds")

DECIMAL_FIGURES = 3


def verb_sem_eval(hyp_lemma, ref_lemma):
    """
    Evaluate hypothesis verbs vs. reference verbs in a SemEval13 style.
    Precision -> 50% dos verbos que estão nos tweets, também estão no tópico
    Recall -> 30% dos verbos que estão no tópico, também estão nos tweets

    :param hyp_lemma: hypothesis verbs lemmatized
    :param ref_lemma: reference verbs lemmatized
    :return: precision and recall of verbs
    """
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

    ##############
    # Topic data #
    ##############

    topic_data = tweetir_data[tweetir_data["topic"] == topic]

    # Data on specific topic
    topic_content = topic_data["topics.content"].unique()[0]
    topic_sentences = tokenize.sent_tokenize(topic_content)

    tweet_multi_doc = topic_data["tweets.full_text"].apply(str.strip).tolist()
    tweets_single_doc = '.\n'.join(tweet_multi_doc)
    tweet_sentences = tokenize.sent_tokenize(tweets_single_doc)

    data_row = {
        "topic": topic, "coref_content": None,
        "tweets_content": tweets_single_doc, "coref_tweets": None, "nr_tweets": len(tweet_multi_doc),
        "topic_clusters": [], "tweets_clusters": [], "topic_sentences_srl": [], "tweets_sentences_srl": [],
        "tweets_verbs": [],
        "metrics": {
            "verb_precision": -1, "verb_recall": -1
        },
        "srl_metrics": {
            "per_verb": {}, "global": {}
        }
    }

    ###########################
    # Co-Reference Resolution #
    ###########################

    print("\nCo-Reference Resolution...")
    start = time.time()

    # Topic
    topic_coref_text, topic_clusters_list = coref_with_lemma(topic_content)
    data_row["coref_content"] = topic_coref_text
    data_row["topic_clusters"] = topic_clusters_list

    # Tweets
    coref_tweets, tweets_clusters_list = coref_with_lemma(tweets_single_doc)
    data_row["coref_tweets"] = coref_tweets
    data_row["tweets_clusters"] = tweets_clusters_list

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")

    #######
    # SRL #
    #######
    # todo: abstract SRL to coref_utils (maybe rename the module)

    print("\nSemantic Role Labelling...")
    start = time.time()

    # Topic
    topic_coref_sentences = tokenize.sent_tokenize(topic_coref_text)
    topic_verbs = []
    topic_results = []
    for topic_sentence in topic_coref_sentences:
        result = predictor.predict_json(
            {"sentence": topic_sentence}
        )

        topic_results.append(result)
        topic_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        topic_verbs.append(verbs)
        # pprint(topic_sent, indent=2)
        data_row["topic_sentences_srl"].append(topic_sent)

    # Topic lemmatization
    topic_verbs_lemma = make_verbs_lemma(topic_verbs)

    # Tweets
    tweets_coref_sentences = tokenize.sent_tokenize(coref_tweets)
    tweets_verbs = []
    tweets_results = []
    for tweet_sentence in tweets_coref_sentences:
        result = predictor.predict_json(
            {"sentence": tweet_sentence}
        )

        tweets_results.append(result)
        tweet_sent = [frame["description"] for frame in result["verbs"]]
        verbs = [frame["verb"] for frame in result["verbs"]]
        tweets_verbs.append(verbs)
        # pprint(tweet_sent, indent=2)
        data_row["tweets_sentences_srl"].append(tweet_sent)

    # Tweets lemmatization
    tweets_verbs_lemma = make_verbs_lemma(tweets_verbs)
    data_row["tweets_verbs"] = NoIndent(tweets_verbs_lemma)

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")

    ##############
    # Evaluation #
    ##############

    print("\nEvaluation...")
    start = time.time()

    # Verb evaluation
    verb_precision, verb_recall = verb_sem_eval(tweets_verbs_lemma, topic_verbs_lemma)
    data_row["metrics"]["verb_precision"] = verb_precision
    data_row["metrics"]["verb_recall"] = verb_recall

    print(f"\nTopic vs Tweets verb recall is {verb_recall} %")

    # SRL Evaluation (compare arguments)
    common_verbs = list(set(tw_verb for tw_verb in tweets_verbs_lemma if tw_verb in topic_verbs_lemma))
    srl_eval_schema = {
        word: {
            "all_frames": {"rouge1_precision": 0, "rouge1_recall": 0, "rouge1_f1": 0},
            "best_frames": {"best_precision": {"rouge1_precision": 0}, "best_recall": {"rouge1_recall": 0},
                            "best_f1": {"rouge1_f1": 0}}
        } for word in common_verbs
    }

    common_verbs_count = {word: 0 for word in common_verbs}
    global_precision, global_recall, global_f1, global_count = 0, 0, 0, 0

    # for each tweet srl prediction result (each sentence)
    for tweet_result in tweets_results:
        # For each frame of the srl performed on the sentence
        for tweet_frame in tweet_result["verbs"]:
            tweet_verb = tweet_frame["verb"]

            best_f1, best_precision = 0, 0
            # Only continue if the verb is also in the topic
            if tweet_verb in common_verbs:
                # For each topic srl prediction result (each sentence)
                for topic_result in topic_results:
                    # For each frame of the srl performed on the sentence
                    for topic_frame in topic_result["verbs"]:
                        topic_verb = topic_frame["verb"]

                        # Find the verb in common between the tweets and the topic
                        if tweet_verb == topic_verb:
                            rouge_s = rouge_scorer.RougeScorer(["rouge1"])

                            non_arg_tags = ["B-V", "I-V", "O"]
                            # topic
                            indexes = value_indexes_in_list(topic_frame["tags"], non_arg_tags, negative_condition=True)
                            w_list = multiple_index_list(topic_result["words"], indexes)
                            if len(w_list) > 0:
                                topic_args = trim_whitespaces(w_list)
                            else:
                                topic_args = ""

                            # tweets
                            indexes = value_indexes_in_list(tweet_frame["tags"], non_arg_tags, negative_condition=True)
                            w_list = multiple_index_list(tweet_result["words"], indexes)
                            if len(w_list) > 0:
                                tweet_args = trim_whitespaces(w_list)
                            else:
                                tweet_args = ""

                            scores = rouge_s.score(topic_args, tweet_args)
                            # Frame with best Rouge1 f-score
                            if scores["rouge1"].fmeasure > srl_eval_schema[tweet_verb]["best_frames"]["best_f1"]["rouge1_f1"]:
                                best_f1_row = {
                                    "topic_args": topic_frame["description"], "tweet_args": tweet_frame["description"],
                                    "rouge1_recall": round(scores["rouge1"].recall, DECIMAL_FIGURES),
                                    "rouge1_precision": round(scores["rouge1"].precision, DECIMAL_FIGURES),
                                    "rouge1_f1": round(scores["rouge1"].fmeasure, DECIMAL_FIGURES)
                                }
                                srl_eval_schema[tweet_verb]["best_frames"]["best_f1"] = best_f1_row

                            # Frame with best Rouge1 precision
                            if scores["rouge1"].precision > srl_eval_schema[tweet_verb]["best_frames"]["best_precision"]["rouge1_precision"]:
                                best_precision_row = {
                                    "topic_args": topic_frame["description"], "tweet_args": tweet_frame["description"],
                                    "rouge1_recall": round(scores["rouge1"].recall, DECIMAL_FIGURES),
                                    "rouge1_precision": round(scores["rouge1"].precision, DECIMAL_FIGURES),
                                    "rouge1_f1": round(scores["rouge1"].fmeasure, DECIMAL_FIGURES)
                                }
                                srl_eval_schema[tweet_verb]["best_frames"]["best_precision"] = best_precision_row

                                # Frame with best Rouge1 recall
                                if scores["rouge1"].recall > srl_eval_schema[tweet_verb]["best_frames"]["best_recall"]["rouge1_recall"]:
                                    best_recall_row = {
                                        "topic_args": topic_frame["description"],
                                        "tweet_args": tweet_frame["description"],
                                        "rouge1_recall": round(scores["rouge1"].recall, DECIMAL_FIGURES),
                                        "rouge1_precision": round(scores["rouge1"].precision, DECIMAL_FIGURES),
                                        "rouge1_f1": round(scores["rouge1"].fmeasure, DECIMAL_FIGURES)
                                    }
                                    srl_eval_schema[tweet_verb]["best_frames"]["best_recall"] = best_recall_row

                            srl_eval_schema[tweet_verb]["all_frames"]["rouge1_precision"] += scores["rouge1"].precision
                            srl_eval_schema[tweet_verb]["all_frames"]["rouge1_recall"] += scores["rouge1"].recall
                            srl_eval_schema[tweet_verb]["all_frames"]["rouge1_f1"] += scores["rouge1"].fmeasure

                            global_precision += scores["rouge1"].precision
                            global_recall += scores["rouge1"].recall
                            global_f1 += scores["rouge1"].fmeasure

                            common_verbs_count[tweet_verb] += 1
                            global_count += 1

    for word, count in zip(common_verbs, common_verbs_count.values()):
        for metric in ["rouge1_precision", "rouge1_recall", "rouge1_f1"]:
            srl_eval_schema[word]["all_frames"][metric] = round(srl_eval_schema[word]["all_frames"][metric] / count, 3)
            srl_eval_schema[word]["all_frames"]["frequency"] = count

    srl_global_metrics = {"rouge1_precision": round(global_precision / global_count, 3),
                          "rouge1_recall": round(global_recall / global_count, 3),
                          "rouge1_f1": round(global_f1 / global_count, 3)}

    data_row["srl_metrics"]["per_verb"] = srl_eval_schema
    data_row["srl_metrics"]["global"] = srl_global_metrics

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds")

    results_list.append(data_row)

    print("\nExporting to JSON...")
    with open(results_dir + "srl_coref_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))
