# Essentials
import os
import pandas as pd
import time
import re
# import json

# Custom modules
from T2S.src.utils import drs_parser as parser

# Cluster similarity
from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings

# from polyfuzz.models import EditDistance
# from jellyfish import jaro_winkler_similarity
# from polyfuzz.models import RapidFuzz

# word processing
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

drs_files_path = os.getcwd() + "/../../datafiles/drs_files/"


def process_clusters(clusters_dict):
    # Divide sentence by words
    clusters_dict = {key: word_tokenize(val) for key, val in clusters_dict.items()}
    # Remove stopwords
    clusters_dict = {key: [word for word in val if not word.lower() in stopwords.words("english")] for key, val in clusters_dict.items()}
    # Join the sentence back and split by actors (",")
    clusters_dict = {key: ' '.join(val).split(", ") for key, val in clusters_dict.items()}
    # Strip whitespaces from actors
    clusters_dict = {key: [text.strip() for text in val] for key, val in clusters_dict.items()}
    # Remove punctuation (from removed stopwords)
    clusters_dict = {key: [''.join(c for c in s if c not in string.punctuation) for s in val] for key, val in
                     clusters_dict.items()}
    # Final whitespace stripping and removing empty words
    clusters_dict = {key: [s.strip() for s in val if s] for key, val in clusters_dict.items()}

    return clusters_dict


if __name__ == '__main__':
    drs_file = drs_files_path + "StadiumTweets_drs.txt"
    t_actors, t_non_ev_rels, t_ev_rels = parser.get_graph_data(drs_file)
    tweets_kg = pd.DataFrame(t_ev_rels, columns=["edge1", "node", "edge2"])
    drs_file = drs_files_path + "StadiumNews_drs.txt"
    n_actors, n_non_ev_rels, n_ev_rels = parser.get_graph_data(drs_file)
    news_kg = pd.DataFrame(n_ev_rels, columns=["edge1", "node", "edge2"])

    print("NEWS - CLUSTERS:")
    print(n_actors)
    print(len(n_actors))
    print("\nTWEETS - CLUSTERS:")
    print(t_actors)
    print(len(t_actors))

    # Remove stopwords
    print("\n1. REMOVING STOPWORDS...")
    # Stopwords processing pipeline:
    n_actors = process_clusters(n_actors)
    t_actors = process_clusters(t_actors)

    # Remove clusters/edges composed only of stopwords
    stopwords_actors = [key for key, val in t_actors.items() if len(val) == 0]
    [t_actors.pop(key, None) for key in stopwords_actors]
    tweets_kg = tweets_kg[(~tweets_kg["edge1"].isin(stopwords_actors)) & (~tweets_kg["edge2"].isin(stopwords_actors))]

    stopwords_actors = [key for key, val in n_actors.items() if len(val) == 0]
    [n_actors.pop(key, None) for key in stopwords_actors]
    news_kg = news_kg[(~news_kg["edge1"].isin(stopwords_actors)) & (~news_kg["edge2"].isin(stopwords_actors))]

    print("NEWS:")
    print(n_actors)
    print(f"Total clusters - {len(n_actors)}")
    print("\nTWEETS:")
    print(t_actors)
    print(f"Total clusters - {len(t_actors)}")

    print("\n2. WHO HAS MORE CLUSTERS? (ignoring for now)")
    # if len(n_actors) >= len(t_actors):
    #     print("The news (>=)")
    #     more_clusters = n_actors
    #     less_clusters = t_actors
    # else:
    #     print("The tweets")
    #     more_clusters = t_actors
    #     less_clusters = n_actors

    print("\n3. CLUSTER SIMILARITY...")
    start = time.time()
    embeddings = TransformerWordEmbeddings("bert-base-multilingual-cased")
    bert = Embeddings(embeddings, min_similarity=0, model_id="BERT")
    model = PolyFuzz(bert)

    # A different model - jaro winkler similarity
    # jellyfish_matcher = EditDistance(scorer=jaro_winkler_similarity)
    # model_jaro_winkler = PolyFuzz(jellyfish_matcher)
    #
    # matcher = RapidFuzz(n_jobs=1)
    # model_rapid = PolyFuzz(matcher)

    paired_clusters, non_paired_clusters = [], []
    for news_key, news_cluster in n_actors.items():
        similar_clusters = []
        for tweets_key, tweets_cluster in t_actors.items():
            model.match(news_cluster, tweets_cluster)
            matches = model.get_matches()

            similar_cases = matches[matches["Similarity"] > 0.8]  # Similarity threshold
            most_similar = matches[matches["Similarity"] == max(matches["Similarity"])]
            if similar_cases.shape[0] != 0:
                row = {
                    "tweets_key": tweets_key, "tweets_cluster": tweets_cluster,
                    "similar_cases": similar_cases.shape[0], "similarity_sum": similar_cases["Similarity"].sum(),
                    "news_representative": most_similar["From"].iloc[0], "tweets_representative": most_similar["To"].iloc[0]
                }
                similar_clusters.append(row)

        if len(similar_clusters) != 0:
            similar_clusters_df = pd.DataFrame(similar_clusters)
            similar_clusters_df.sort_values(["similar_cases", "similarity_sum"], ascending=False, inplace=True)
            similar_cluster = similar_clusters_df.iloc[0]
            paired_clusters.append({
                "news_key": news_key, "news_cluster": news_cluster,
                "news_cluster_representative": similar_cluster["news_representative"],
                "tweets_key": similar_cluster["tweets_key"], "tweets_cluster": similar_cluster["tweets_cluster"],
                "tweets_cluster_representative": similar_cluster["tweets_representative"]
            })
        else:
            non_paired_clusters.append({"news_key": news_key, "news_cluster": news_cluster})

    paired_clusters = pd.DataFrame(paired_clusters)
    non_paired_clusters = pd.DataFrame(non_paired_clusters)
    non_paired_clusters["news_cluster_representative"] = non_paired_clusters["news_cluster"].apply(lambda x: x[0])

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds\n")

    print("\nSimilar clusters (news vs. tweets):")
    print(paired_clusters)
    print("\nClusters with no similar pair (only news):")
    print(non_paired_clusters)

    print("\n4. MAPPING ACTORS TO CLUSTER VALUES...")
    # News mapping actors to cluster representative
    temp1 = paired_clusters.drop(paired_clusters.columns[[1, 3, 4, 5]], axis=1)
    temp2 = non_paired_clusters.drop(non_paired_clusters.columns[1], axis=1)
    news_mapping = temp1.append(temp2, ignore_index=True)

    # Tweets mapping actors to cluster representative
    temp1 = paired_clusters.drop(paired_clusters.columns[[0, 1, 2, 4]], axis=1)
    keys = [key for key in t_actors.keys() if key not in temp1["tweets_key"].to_list()]
    temp2 = pd.DataFrame.from_dict({key: t_actors[key][0] for key in keys}, orient="index").reset_index()
    temp2.columns = ["tweets_key", "tweets_cluster_representative"]
    tweets_mapping = temp1.append(temp2, ignore_index=True)

    print("\nNEWS:")
    print(news_mapping)
    print("\nTWEETS:")
    print(tweets_mapping)

    news_kg.insert(1, "Redge1",
                   news_kg.apply(lambda x: news_mapping[news_mapping["news_key"] == x["edge1"]].iloc[0, 1], axis=1))
    news_kg.insert(3, "Redge2",
                   news_kg.apply(lambda x: news_mapping[news_mapping["news_key"] == x["edge2"]].iloc[0, 1], axis=1))

    news_kg.insert(3, "node1", news_kg.apply(lambda x: re.sub("[\(].*?[\)]", "", x["node"]).strip(), axis=1))
    news_kg.insert(4, "node2", news_kg.apply(lambda x: x["node"][x["node"].find("(") + 1:x["node"].find(")")], axis=1))

    tweets_kg.insert(
        1, "Redge1",
        tweets_kg.apply(lambda x: tweets_mapping[tweets_mapping["tweets_key"] == x["edge1"]].iloc[0, 1], axis=1)
    )

    tweets_kg.insert(
        3, "Redge2",
        tweets_kg.apply(lambda x: tweets_mapping[tweets_mapping["tweets_key"] == x["edge2"]].iloc[0, 1], axis=1)
    )

    tweets_kg.insert(3, "node1", tweets_kg.apply(lambda x: re.sub("[\(].*?[\)]", "", x["node"]).strip(), axis=1))
    tweets_kg.insert(4, "node2", tweets_kg.apply(lambda x: x["node"][x["node"].find("(") + 1:x["node"].find(")")], axis=1))

    print("\n5. COMPARE TRIPLES...")
    print("\nExisting triples...")
    print("NEWS:")
    print(news_kg[["Redge1", "node1", "Redge2"]])
    print("\nTWEETS:")
    print(tweets_kg[["Redge1", "node1", "Redge2"]])

    start = time.time()
    paired_KG_evaluation = pd.DataFrame()
    for cluster in paired_clusters.itertuples():
        news_triples = news_kg[news_kg["edge1"] == cluster.news_key]
        tweets_triples = tweets_kg[tweets_kg["edge1"] == cluster.tweets_key]
        for tnews in news_triples.itertuples():
            tnews_scores = []
            for ttweets in tweets_triples.itertuples():
                triple_similarity = 0
                # print("\nPar de triples (news - tweets) para avaliação:")
                # print(f"News triple: {tnews.Redge1} - {tnews.node1} - {tnews.Redge2}")
                # print(f"News triple: {ttweets.Redge1} - {ttweets.node1} - {ttweets.Redge2}")

                triple_similarity += model.match([tnews.Redge1], [ttweets.Redge1]).get_matches().loc[0, "Similarity"]
                triple_similarity += model.match([tnews.node1], [ttweets.node1]).get_matches().loc[0, "Similarity"]
                triple_similarity += model.match([tnews.Redge2], [ttweets.Redge2]).get_matches().loc[0, "Similarity"]
                triple_similarity = round(triple_similarity / 3, 3)
                tnews_scores.append({"news_triple": f"{tnews.Redge1} - {tnews.node1} - {tnews.Redge2}",
                                     "best_tweets_triple": f"{ttweets.Redge1} - {ttweets.node1} - {ttweets.Redge2}",
                                     "bert_similarity": triple_similarity})

            df = pd.DataFrame(tnews_scores)
            best_result = df[df["bert_similarity"] == max(df["bert_similarity"])].iloc[0]
            paired_KG_evaluation = paired_KG_evaluation.append(best_result, ignore_index=True)

    print("\nPAIRED CLUSTERS TRIPLES EVALUATION...")
    paired_KG_evaluation = paired_KG_evaluation[["news_triple", "best_tweets_triple", "bert_similarity"]]
    print(paired_KG_evaluation)

    end = time.time()
    print(f"\nComputation time - {round(end - start, 2)} seconds")

    # non paired KG evaluation (triples da notícia para os quais não encontro semelhantes nos tweets)
    print("\nNON PAIRED CLUSTERS TRIPLES EVALUATION...")
    non_paired_KG = news_kg[~news_kg["edge1"].isin(paired_clusters["news_key"])]

    non_paired_KG_evaluation = pd.DataFrame()
    for news_triple in non_paired_KG.itertuples():
        non_paired_score = []
        for ttweets in tweets_kg.itertuples():
            triple_similarity = 0
            triple_similarity += model.match([news_triple.Redge1], [ttweets.Redge1]).get_matches().loc[0, "Similarity"]
            triple_similarity += model.match([news_triple.node1], [ttweets.node1]).get_matches().loc[0, "Similarity"]
            triple_similarity += model.match([news_triple.Redge2], [ttweets.Redge2]).get_matches().loc[0, "Similarity"]
            triple_similarity = round(triple_similarity / 3, 3)
            non_paired_score.append({"news_triple": f"{news_triple.Redge1} - {news_triple.node1} - {news_triple.Redge2}",
                                 "best_tweets_triple": f"{ttweets.Redge1} - {ttweets.node1} - {ttweets.Redge2}",
                                 "bert_similarity": triple_similarity})

        df = pd.DataFrame(non_paired_score)
        best_result = df[df["bert_similarity"] == max(df["bert_similarity"])].iloc[0]
        non_paired_KG_evaluation = non_paired_KG_evaluation.append(best_result, ignore_index=True)

    non_paired_KG_evaluation = non_paired_KG_evaluation[["news_triple", "best_tweets_triple", "bert_similarity"]]
    print(non_paired_KG_evaluation)

    print("\n6. FINAL KG EVALUATION...")
    final_evaluation = paired_KG_evaluation.append(non_paired_KG_evaluation, ignore_index=True)
    print(final_evaluation)
    print(f"\nMEAN SIMILARITY BETWEEN NEWS AND TWEETS TRIPLES - {round(final_evaluation['bert_similarity'].mean(), 3)}")
