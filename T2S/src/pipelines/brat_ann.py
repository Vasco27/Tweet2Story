# Essentials
import json
import re
from collections import namedtuple

# Custom modules
from T2S.src.utils.data_utils import get_paths
from T2S.src.utils.json_utils import NoIndent, MyEncoder
from T2S.src.utils.string_utils import trim_whitespaces, normalize_entities

# Spacy - Named-Entity Recognition
import en_core_web_trf
nlp = en_core_web_trf.load()

if __name__ == '__main__':
    # This is a test, using a test file with only one topic (the Lydia Ko golfing article)
    path_to_data, results_dir = get_paths()
    srl_dir = results_dir + "baselines/SRL/"
    tt_dir = results_dir + "baselines/TT/"
    ann_dir = results_dir + "annotations/"

    with open(srl_dir + "srl_coref_exp.json") as f:
        srl_data = json.load(f)[0]

    with open(tt_dir + "tt_exp4.json") as f:
        tt_data = json.load(f)[0]

    results_list = []
    data_row = {
        "coref_annotations": None, "time_annotations": None, "remaining_annotations": None
    }

    #######################################
    # Co-Reference Resolution annotations #
    #######################################

    doc_coref_tweets = nlp(srl_data["coref_tweets"])
    tweets_coref_ner = [(X.text, X.label_) for X in doc_coref_tweets.ents]

    doc_tweets = nlp(srl_data["tweets_content"])
    tweets_ner = [(X.text, X.label_) for X in doc_tweets.ents]

    clusters_annotations, clusters_ents, clusters_labels = [], [], []
    for cluster in srl_data["tweets_clusters"]:
        # Make a cluster set (eliminate duplicated corefs) ordered by the most common coref in the cluster
        cluster_set = sorted(set(cluster), key=cluster.count)[::-1]
        cluster_set = [trim_whitespaces(set_value) for set_value in cluster_set]

        # remove spaces around hyphens
        cluster_set = [re.sub(r'[ ](?=-)|(?<=-)[ ]', '', set_value) for set_value in cluster_set]
        clusters_ents.append(cluster_set)

        ent_found = False
        for cluster_value in cluster_set:

            # Check if the tweets NER after applying the coref has a match for the cluster entity.
            cluster_ner = [label for ent, label in tweets_coref_ner if ent.lower() == cluster_value.lower()]
            if len(cluster_ner) > 0:
                clusters_annotations.append((NoIndent(cluster_set), cluster_ner[0]))
                clusters_labels.append(cluster_ner[0])
                ent_found = True
                break

            # Check if the tweets NER without coref has a match for the cluster entity
            cluster_ner = [label for ent, label in tweets_ner if ent.lower() == cluster_value.lower()]
            if len(cluster_ner) > 0:
                clusters_annotations.append((NoIndent(cluster_set), cluster_ner[0]))
                clusters_labels.append(cluster_ner[0])
                ent_found = True
                break

            # Check if the cluster value has a verb to classify the cluster entity as an event
            if any(word in srl_data["tweets_verbs"] for word in cluster_value.split()):
                clusters_annotations.append((NoIndent(cluster_set), "EVENT"))
                clusters_labels.append("EVENT")
                ent_found = True
                break

        # todo: default value for when we can't classify the coref entity?
        if not ent_found:
            clusters_annotations.append((NoIndent(cluster_set), ""))
            clusters_labels.append("")

    data_row["coref_annotations"] = clusters_annotations

    ####################
    # Date annotations #
    ####################

    time_refs = tt_data["tt_tweets"]
    [t_ref.append("DATE") for t_ref in time_refs]

    data_row["time_annotations"] = time_refs

    time_ents = [t_ref[1] for t_ref in time_refs]
    clusters_ents.append(time_ents)

    #####################
    # Other annotations #
    #####################
    normalized_ents = normalize_entities(clusters_ents, nested_list=True)

    # remaining_ents = [(ent, label) for ent, label in tweets_ner if ent.lower() not in clusters_ents]
    remaining_ents = []
    for ent, label in tweets_ner:
        # Normalize the entity - lower, remove punctuation and split the word
        normalized_ent = re.sub(r"[^\w\s]", "", ent.lower()).split()

        # Check if the extracted entity was already classified by the coref/tt clusters
        if not any(word for word in normalized_ent if word in normalized_ents):
            remaining_ents.append((ent, label))

    data_row["remaining_annotations"] = remaining_ents

    results_list.append(data_row)

    #########################
    # Make brat file (.ann) #
    #########################

    mod_content = srl_data["tweets_content"].lower()
    text_bound_id = 1
    ann = ""

    # Algorithm for retrieving start and end char positions of named entities
    # Starting with the coref clusters entities
    for cluster, label in zip(srl_data["tweets_clusters"], clusters_labels):
        cluster_data = [trim_whitespaces(set_value) for set_value in cluster]

        # remove spaces around hyphens and parenthesis
        re_special_ws = r'([ ](?=-)|(?<=-)[ ])|([ ](?=\))|(?<=\()[ ])'
        cluster_data = [re.sub(re_special_ws, '', value.lower()) for value in cluster_data]

        for cluster_value in cluster_data:
            text_bound_id_str = "T" + str(text_bound_id)
            cluster_value = cluster_value.translate(str.maketrans({"-": r"\-", "]": r"\]", "^": r"\^",
                                                                   "$": r"\$", "*": r"\*", ".": r"\.",
                                                                   "(": r"\(", ")": r"\)"}))
            c = re.search(r'({}|$)'.format(cluster_value), mod_content)
            start_char = c.start()
            end_char = c.end()

            str_size = end_char - start_char
            str_to_replace = "X" * str_size

            # Modify the content to replace entity with placeholder
            mod_content = mod_content.replace(cluster_value, str_to_replace, 1)

            ann = ann + f"{text_bound_id_str}\t{label} {start_char} {end_char}\t{cluster_value}\n"
            text_bound_id += 1

    ######################
    # Export annotations #
    ######################

    with open(ann_dir + "tweets_ann_exp.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(results_list, cls=MyEncoder, ensure_ascii=False, indent=4))

    with open(ann_dir + "5811057c-6732-4b37-b04c-ddf0a75a7b51.ann", "w", encoding="utf-8") as f:
        f.write(ann)
