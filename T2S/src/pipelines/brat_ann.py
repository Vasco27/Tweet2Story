# Essentials
import json

# Custom modules
from T2S.src.utils.data_utils import get_paths

# Spacy - Named-Entity Recognition
import en_core_web_trf
nlp = en_core_web_trf.load()

if __name__ == '__main__':
    # This is a test, using a test file with only one topic (the Lydia Ko golfing article)
    path_to_data, results_dir = get_paths()
    srl_dir = results_dir + "baselines/SRL/"
    tt_dir = results_dir + "baselines/TT/"

    with open(srl_dir + "srl_coref_exp.json") as f:
        srl_data = json.load(f)[0]

    with open(tt_dir + "tt_exp4.json") as f:
        tt_data = json.load(f)[0]

    doc_coref_tweets = nlp(srl_data["coref_tweets"])
    tweets_coref_ner = [(X.text, X.label_) for X in doc_coref_tweets.ents]

    doc_tweets = nlp(srl_data["tweets_content"])
    tweets_ner = [(X.text, X.label_) for X in doc_tweets.ents]

    clusters_annotations = []
    for cluster in srl_data["tweets_clusters"]:
        cluster_set = sorted(set(cluster), key=cluster.count)[::-1]
        for cluster_value in cluster_set:
            cluster_ner = [label for ent, label in tweets_coref_ner if ent.lower() == cluster_value.lower()]
            if len(cluster_ner) > 0:
                clusters_annotations.append((cluster_set, cluster_ner[0]))
                break

            cluster_ner = [label for ent, label in tweets_ner if ent.lower() == cluster_value.lower()]
            if len(cluster_ner) > 0:
                clusters_annotations.append((cluster_set, cluster_ner[0]))
                break

            if any(word in srl_data["tweets_verbs"] for word in cluster_value.split()):
                clusters_annotations.append((cluster_set, "EVENT"))
                break

    print(clusters_annotations)
