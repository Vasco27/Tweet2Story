# Essentials
import pandas as pd
import json

# Custom utils
from T2S.src.utils.data_utils import get_paths
from T2S.src.utils.eval_utils import ner_confusion_matrix
from T2S.src.utils.json_utils import MyEncoder, NoIndent

# Spacy
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


if __name__ == '__main__':
    path_to_data, results_dir = get_paths()
    results_dir = results_dir + "baselines/NER/"

    tweetir_data = pd.read_csv(path_to_data)
    topic_ids = tweetir_data["topic"].unique()

    topic_docs = tweetir_data["topics.content"].unique().tolist()

    iteration, total_iterations, no_entities = 1, tweetir_data["topic"].unique().shape[0], 0
    results_list = []
    # for topic in tweetir_data["topic"].unique():
    topic = "5811057c-6732-4b37-b04c-ddf0a75a7b51"

    data_row = {
        "topic": topic, "content": topic_docs[24],
        "ner_topic": [], "ner_tweets": []
    }

    # start = time.time()
    print(f"Iteration {iteration} of {total_iterations}")
    topic_data = tweetir_data[tweetir_data["topic"] == topic]

    # Data on specific topic
    topics_content = topic_data["topics.content"].unique()
    tweet_multi_doc = topic_data["tweets.full_text"].tolist()
    tweets_single_doc = '\n'.join(tweet_multi_doc)

    # Topic entity extraction
    doc_topic = nlp(topics_content[0])
    topic_entities = process_entity_list(doc_topic.ents)
    topic_ents_labels = process_entity_label_list(doc_topic.ents)
    data_row["ner_topic"] = NoIndent(list(topic_entities))
    # pprint(topic_ents_labels)

    # Tweets entity extraction
    doc_tweets = nlp(tweets_single_doc)
    tweets_entities = process_entity_list(doc_tweets.ents)
    tweets_ents_labels = process_entity_label_list(doc_tweets.ents)
    data_row["ner_tweets"] = NoIndent(list(tweets_entities))
    # pprint(tweets_ents_labels)

    # Entities with duplicates (for weighted recall)
    t_ents = [X.text.lower() for X in doc_topic.ents]
    tw_ents = [X.text.lower() for X in doc_tweets.ents]

    # EVALUATION
    eval_schema = ner_confusion_matrix(topic_ents_labels, tweets_ents_labels)
    # todo: eval_utils function for computing confusion matrix metrics
    COR = sum(eval_schema["metrics"][measure]["COR"] for measure in ["strict", "exact", "partial", "type"])
    INC = sum(eval_schema["metrics"][measure]["INC"] for measure in ["strict", "exact", "partial", "type"])
    PAR = sum(eval_schema["metrics"][measure]["PAR"] for measure in ["strict", "exact", "partial", "type"])
    MIS = sum(eval_schema["metrics"][measure]["MIS"] for measure in ["strict", "exact", "partial", "type"])
    SPU = sum(eval_schema["metrics"][measure]["SPU"] for measure in ["strict", "exact", "partial", "type"])

    POS = COR + INC + PAR + MIS  # TP + FN
    ACT = COR + INC + PAR + SPU  # TP + FP

    # Exact match eval
    exact_precision = COR / ACT
    exact_recall = COR / POS
    exact_F1 = (2 * exact_precision * exact_recall) / (exact_precision + exact_recall)

    # Partial match eval
    partial_precision = (COR + 0.5 * PAR) / ACT  # TP / (TP + FP)
    partial_recall = (COR + 0.5 * PAR) / POS  # TP / (TP + FP)
    partial_F1 = (2 * partial_precision * partial_recall) / (partial_precision + partial_recall)

    metrics_schema = {
        "exact_precision": round(exact_precision * 100, DECIMAL_FIGURES),
        "exact_recall": round(exact_recall * 100, DECIMAL_FIGURES),
        "exact_F1": round(exact_F1 * 100, DECIMAL_FIGURES),
        "partial_precision": round(partial_precision * 100, DECIMAL_FIGURES),
        "partial_recall": round(partial_recall * 100, DECIMAL_FIGURES),
        "partial_F1": round(partial_F1 * 100, DECIMAL_FIGURES)
    }

    data_row = {**data_row, **metrics_schema}
    results_list.append(data_row)

    with open(results_dir + "ner_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))

    # iteration += 1
    # end = time.time()
    # print(f"Iteration time - {round(end - start, 2)} seconds.")
