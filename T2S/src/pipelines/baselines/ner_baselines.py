# Essentials
import pandas as pd
import json
from collections import namedtuple
import time

# Custom utils
from T2S.src.utils.data_utils import get_paths
from T2S.src.utils.eval_utils import semeval_confusion_matrix, semeval_metrics_computation
from T2S.src.utils.json_utils import MyEncoder, NoIndent

# Named-Entity Recognition
# from spacy import displacy  # Used for visualization (works better on notebooks)
import en_core_web_trf
nlp = en_core_web_trf.load()

DECIMAL_FIGURES = 3


def process_entity_list(ents):
    ent_list = [X.text for X in ents]

    ent_list = set(map(str, ent_list))
    ent_list = set(map(str.strip, ent_list))
    ent_list = set(map(str.lower, ent_list))

    return ent_list


def process_entity_label_list(ents):
    ent_label_list = [(X.text, X.label_) for X in ents]

    ent_label_list = [tuple(map(str, tuple1)) for tuple1 in set(ent_label_list)]
    ent_label_list = [tuple(map(str.strip, tuple1)) for tuple1 in set(ent_label_list)]
    ent_label_list = [tuple(map(str.lower, tuple1)) for tuple1 in set(ent_label_list)]

    return set(ent_label_list)


def compute_ner_weighted_recall(ents_labels, text, ref_text):
    """
    Computes weighted recall according to the formula on "keyword_ext_baselines.py" method "nr_keywords_in_text".
    Generates a list of named tuples that represent Named-Entities in the form of (entity, label, weighted_recall).

    :param ents_labels: list of (entity, label) tuples extracted from the NER baseline
    :param text: the document from where to calculate the importance of the word
    :param ref_text: the document from where the keywords are extracted
    :return: a list of Named-Entities in namedTuple form

    Note: The weighted recall is a measure of word importance in a text that considers its frequencies in both text
    and ref_text.
    """
    ner_list = []
    low_text = text.lower()
    low_ref_text = ref_text.lower()

    text_freqs = [low_text.count(ent) for (ent, label) in ents_labels]
    total_ent_freq = sum(text_freqs)

    text_ref_freqs = [low_ref_text.count(ent) for (ent, label) in ents_labels]
    total_ent_ref_freq = sum(text_ref_freqs)

    for (ent, label), freq, ref_freq in zip(ents_labels, text_freqs, text_ref_freqs):
        if total_ent_freq != 0:
            weighted_recall = round((freq + ref_freq) / (total_ent_freq + total_ent_ref_freq), DECIMAL_FIGURES)
        else:
            weighted_recall = 0

        ner_list.append(NamedEntity(entity=ent, label=label, w_recall=weighted_recall)._asdict())

    return ner_list


if __name__ == '__main__':
    path_to_data, results_dir = get_paths()
    results_dir = results_dir + "baselines/NER/"

    tweetir_data = pd.read_csv(path_to_data)
    topic_ids = tweetir_data["topic"].unique()

    topic_docs = tweetir_data["topics.content"].unique().tolist()

    # named tuple for keyword representation in json
    NamedEntity = namedtuple("NE", ["entity", "label", "w_recall"])

    iteration, total_iterations, no_entities = 1, tweetir_data["topic"].unique().shape[0], 0
    results_list = []
    for topic in topic_ids:
        # topic = "5811057c-6732-4b37-b04c-ddf0a75a7b51"
        start = time.time()

        # start = time.time()
        print(f"Iteration {iteration} of {total_iterations}")
        topic_data = tweetir_data[tweetir_data["topic"] == topic]

        # Data on specific topic
        topic_content = topic_data["topics.content"].unique()[0]
        tweet_multi_doc = topic_data["tweets.full_text"].tolist()
        tweets_single_doc = '\n'.join(tweet_multi_doc)

        data_row = {
            "topic": topic, "content": topic_content, "nr_tweets": len(tweet_multi_doc),
            "ner_topic": [], "ner_tweets": []
        }

        # Topic entity extraction
        doc_topic = nlp(topic_content)
        topic_ents = process_entity_list(doc_topic.ents)
        topic_ents_labels = process_entity_label_list(doc_topic.ents)
        # data_row["ner_topic"] = NoIndent(list(topic_ents))
        # pprint(topic_ents_labels)

        # Tweets entity extraction
        doc_tweets = nlp(tweets_single_doc)
        tweets_ents_labels = process_entity_label_list(doc_tweets.ents)
        # tweets_ner_list = compute_ner_weighted_recall(tweets_ents_labels, topic_content, tweets_single_doc)
        # data_row["ner_tweets"] = NoIndent(tweets_ner_list)
        # pprint(tweets_ents_labels)

        # EVALUATION
        eval_schema = semeval_confusion_matrix(topic_ents_labels, tweets_ents_labels)
        metrics_schema = semeval_metrics_computation(eval_schema, DECIMAL_FIGURES)

        data_row = {**data_row, **metrics_schema}

        results_list.append(data_row)

        iteration += 1
        end = time.time()
        print(f"Iteration time - {round(end - start, 2)} seconds.")

    # with open(results_dir + "ner_baselines.json", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))

    print(sum(ner["exact_precision"] for ner in results_list))
    print(sum(ner["exact_recall"] for ner in results_list))
    print(sum(ner["exact_F1"] for ner in results_list))
    print(sum(ner["partial_precision"] for ner in results_list))
    print(sum(ner["partial_recall"] for ner in results_list))
    print(sum(ner["partial_F1"] for ner in results_list))
