# Essentials
import os
import pandas as pd
import time
import re
from pathlib import Path
import json

# Custom modules
from T2S.src.utils import drs_parser as parser

# Cluster similarity
from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings, RapidFuzz
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings

# Metrics
from rouge_score import rouge_scorer

# word processing
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

drs_files_path = os.path.join(Path(__file__).parent.parent.parent, "datafiles", "drs_files")
results_path = os.path.join(Path(__file__).parent.parent.parent, "resultfiles", "triple_evaluation")

ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1"])


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


def remove_stopwords(actors, kg):
    # Stopwords processing pipeline
    actors = process_clusters(actors)

    # Remove clusters/edges composed only of stopwords
    stopwords_actors = [key for key, val in actors.items() if len(val) == 0]
    [actors.pop(key, None) for key in stopwords_actors]
    kg = kg[(~kg["edge1"].isin(stopwords_actors)) & (~kg["edge2"].isin(stopwords_actors))]

    return actors, kg


def cluster_similarity(model, clusters_from, clusters_to):
    paired_clusters, non_paired_clusters = [], []
    for news_key, news_cluster in clusters_from.items():
        similar_clusters = []
        best_similarity = 0
        row = {}
        for tweets_key, tweets_cluster in clusters_to.items():
            model.match(news_cluster, tweets_cluster)
            matches = model.get_matches()

            similar_cases = matches[matches["Similarity"] > 0.8]  # Similarity threshold
            most_similar = matches[matches["Similarity"] == max(matches["Similarity"])]
            if similar_cases.shape[0] != 0:
                new_similarity = similar_cases["Similarity"].sum()
                if new_similarity > best_similarity:
                    best_similarity = new_similarity
                    # Since we know they are similar, it should be the same for news and tweets
                    representative = most_similar["From"].iloc[0]
                    row = {
                        "tweets_key": tweets_key, "tweets_cluster": tweets_cluster,
                        "similar_cases": similar_cases.shape[0], "similarity_sum": similar_cases["Similarity"].sum(),
                        "news_representative": representative, "tweets_representative": representative
                    }
        if len(row) != 0:
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

    return paired_clusters, non_paired_clusters


def map_news_in_kg(news_kg, pclusters, npclusters):
    # News mapping actors to cluster representative
    temp1 = pclusters.drop(pclusters.columns[[1, 3, 4, 5]], axis=1)
    temp2 = npclusters.drop(npclusters.columns[1], axis=1)
    news_mapping = temp1.append(temp2, ignore_index=True)

    news_kg.insert(1, "Redge1",
                   news_kg.apply(lambda x: news_mapping[news_mapping["news_key"] == x["edge1"]].iloc[0, 1], axis=1))
    news_kg.insert(3, "Redge2",
                   news_kg.apply(lambda x: news_mapping[news_mapping["news_key"] == x["edge2"]].iloc[0, 1], axis=1))

    news_kg.insert(3, "node1", news_kg.apply(lambda x: re.sub("[\(].*?[\)]", "", x["node"]).strip(), axis=1))
    news_kg.insert(4, "node2", news_kg.apply(lambda x: x["node"][x["node"].find("(") + 1:x["node"].find(")")], axis=1))

    return news_kg


def map_tweets_in_kg(tweets_kg, pclusters, actors):
    # Tweets mapping actors to cluster representative
    temp1 = pclusters.drop(pclusters.columns[[0, 1, 2, 4]], axis=1)
    keys = [key for key in actors.keys() if key not in temp1["tweets_key"].to_list()]
    temp2 = pd.DataFrame.from_dict({key: actors[key][0] for key in keys}, orient="index").reset_index()
    temp2.columns = ["tweets_key", "tweets_cluster_representative"]
    tweets_mapping = temp1.append(temp2, ignore_index=True)

    tweets_kg.insert(
        1, "Redge1",
        tweets_kg.apply(lambda x: tweets_mapping[tweets_mapping["tweets_key"] == x["edge1"]].iloc[0, 1], axis=1)
    )

    tweets_kg.insert(
        3, "Redge2",
        tweets_kg.apply(lambda x: tweets_mapping[tweets_mapping["tweets_key"] == x["edge2"]].iloc[0, 1], axis=1)
    )

    tweets_kg.insert(3, "node1", tweets_kg.apply(lambda x: re.sub("[\(].*?[\)]", "", x["node"]).strip(), axis=1))
    tweets_kg.insert(4, "node2",
                     tweets_kg.apply(lambda x: x["node"][x["node"].find("(") + 1:x["node"].find(")")], axis=1))

    return tweets_kg


def triples_evaluation(model, news_triples, tweets_triples, model_names):
    tnews_scores = []
    for tnews in news_triples.itertuples():
        best_similarity = 0
        best_row = {}
        for ttweets in tweets_triples.itertuples():
            triple_similarity = 0

            try:
                triple_similarity += model.match(
                    [tnews.Redge1], [ttweets.Redge1, "aaaaaaaaaaaaaaaaa"]
                ).get_matches("BERT").loc[0, "Similarity"]
                triple_similarity += model.match(
                    [tnews.node1], [ttweets.node1, "aaaaaaaaaaaaaaaaa"]
                ).get_matches("BERT").loc[0, "Similarity"]
                triple_similarity += model.match(
                    [tnews.Redge2], [ttweets.Redge2, "aaaaaaaaaaaaaaaaa"]
                ).get_matches("BERT").loc[0, "Similarity"]
                triple_similarity = round(triple_similarity / 3, 3)
            except Exception as ex:
                print(tnews)
                print(ttweets)
                triple_similarity = 0.5
                print(repr(ex))

            if triple_similarity > best_similarity:
                best_similarity = triple_similarity

                rouge_score = 0
                rouge_score += ROUGE_SCORER.score(tnews.Redge1, ttweets.Redge1)["rouge1"].fmeasure
                rouge_score += ROUGE_SCORER.score(tnews.node1, ttweets.node)["rouge1"].fmeasure
                rouge_score += ROUGE_SCORER.score(tnews.Redge2, ttweets.Redge2)["rouge1"].fmeasure
                rouge_score = round(rouge_score / 3, 3)

                scores = []
                for model_name in model_names[1:]:
                    triple_similarity = 0

                    try:
                        triple_similarity += model.match(
                            [tnews.Redge1], [ttweets.Redge1, "aaaaaaaaaaaaaaaaa"]
                        ).get_matches(model_name).loc[0, "Similarity"]
                        triple_similarity += model.match(
                            [tnews.node1], [ttweets.node1, "aaaaaaaaaaaaaaaaa"]
                        ).get_matches(model_name).loc[0, "Similarity"]
                        triple_similarity += model.match(
                            [tnews.Redge2], [ttweets.Redge2, "aaaaaaaaaaaaaaaaa"]
                        ).get_matches(model_name).loc[0, "Similarity"]
                        triple_similarity = round(triple_similarity / 3, 3)
                    except Exception as ex:
                        print(tnews)
                        print(ttweets)
                        triple_similarity = 0.5
                        print(repr(ex))

                    scores.append(triple_similarity)
                best_row = {"news_triple": f"{tnews.Redge1} - {tnews.node1} - {tnews.Redge2}",
                            "best_tweets_triple": f"{ttweets.Redge1} - {ttweets.node1} - {ttweets.Redge2}",
                            model_names[0]: best_similarity, model_names[1]: scores[0], model_names[2]: scores[1],
                            "rouge1": rouge_score}
        tnews_scores.append(best_row)

    return pd.DataFrame(tnews_scores)


def main(tweets_file, news_file, verbose=True):
    t_actors, t_non_ev_rels, t_ev_rels = parser.get_graph_data(tweets_file)
    tweets_kg = pd.DataFrame(t_ev_rels, columns=["edge1", "node", "edge2"])
    t_actors = {key: val.lower() for key, val in t_actors.items()}

    n_actors, n_non_ev_rels, n_ev_rels = parser.get_graph_data(news_file)
    news_kg = pd.DataFrame(n_ev_rels, columns=["edge1", "node", "edge2"])
    n_actors = {key: val.lower() for key, val in n_actors.items()}

    print("\n1. REMOVING STOPWORDS...")
    n_actors, news_kg = remove_stopwords(n_actors, news_kg)
    t_actors, tweets_kg = remove_stopwords(t_actors, tweets_kg)

    print("\n2. CLUSTER SIMILARITY...")
    start = time.time()

    embeddings = TransformerWordEmbeddings("bert-base-multilingual-cased")
    bert = Embeddings(embeddings, min_similarity=0, model_id="BERT")
    model_bert = PolyFuzz(bert)

    paired_clusters, non_paired_clusters = cluster_similarity(model_bert, n_actors, t_actors)

    end = time.time()
    print(f"Computation time - {round(end - start, 2)} seconds\n")

    if verbose:
        print("\nSimilar clusters (news vs. tweets):")
        print(paired_clusters)
        print("\nClusters with no similar pair (only news):")
        print(non_paired_clusters)

    print("\n3. MAPPING ACTORS TO CLUSTER VALUES...")
    news_kg = map_news_in_kg(news_kg, paired_clusters, non_paired_clusters)
    tweets_kg = map_tweets_in_kg(tweets_kg, paired_clusters, t_actors)

    news_kg["Redge1"] = news_kg["Redge1"].str.lower()
    news_kg["Redge2"] = news_kg["Redge2"].str.lower()
    news_kg["node1"] = news_kg["node1"].str.lower()

    tweets_kg["Redge1"] = tweets_kg["Redge1"].str.lower()
    tweets_kg["Redge2"] = tweets_kg["Redge2"].str.lower()
    tweets_kg["node1"] = tweets_kg["node1"].str.lower()

    if verbose:
        print("\nExisting triples...")
        print("NEWS:")
        # print(news_kg[["Redge1", "node1", "Redge2"]])
        print(news_kg)
        print("\nTWEETS:")
        # print(tweets_kg[["Redge1", "node1", "Redge2"]])
        print(tweets_kg)

    print("\n4. COMPARE TRIPLES...")
    fasttext_embeddings = WordEmbeddings('en-crawl')
    fasttext = Embeddings(fasttext_embeddings, min_similarity=0, model_id="FastText")
    leven_dist = RapidFuzz(n_jobs=1, model_id="leven")

    model_names = ["BERT", "FastText", "leven"]
    models = [bert, fasttext, leven_dist]
    model = PolyFuzz(models)

    start = time.time()
    paired_KG_evaluation = pd.DataFrame()
    for cluster in paired_clusters.itertuples():
        news_triples = news_kg[news_kg["edge1"] == cluster.news_key]
        tweets_triples = tweets_kg[tweets_kg["edge1"] == cluster.tweets_key]
        paired_KG_evaluation = paired_KG_evaluation.append(
            triples_evaluation(model, news_triples, tweets_triples, model_names),
            ignore_index=True)

    if verbose:
        print("\nPAIRED CLUSTERS TRIPLES EVALUATION...")
        print(paired_KG_evaluation)

    end = time.time()
    print(f"\nComputation time - {round(end - start, 2)} seconds")

    # non paired KG evaluation (triples da notícia para os quais não encontro semelhantes nos tweets)
    print("\nNON PAIRED CLUSTERS TRIPLES EVALUATION...")
    non_paired_KG = news_kg[~news_kg["edge1"].isin(paired_clusters["news_key"])]
    non_paired_KG_evaluation = triples_evaluation(model, non_paired_KG, tweets_kg, model_names)
    print(non_paired_KG_evaluation)

    print("\n5. FINAL KG EVALUATION...")
    final_evaluation = paired_KG_evaluation.append(non_paired_KG_evaluation, ignore_index=True)
    print(final_evaluation)
    final_leven = round(final_evaluation['leven'].mean(), 3)
    final_fasttext = round(final_evaluation['FastText'].mean(), 3)
    final_bert = round(final_evaluation['BERT'].mean(), 3)
    final_rouge1 = round(final_evaluation['rouge1'].mean(), 3)
    print(f"\nMEAN LEVEN SIMILARITY BETWEEN NEWS AND TWEETS TRIPLES - {final_leven}")
    print(f"\nMEAN FAST TEXT SIMILARITY BETWEEN NEWS AND TWEETS TRIPLES - {final_fasttext}")
    print(f"\nMEAN BERT SIMILARITY BETWEEN NEWS AND TWEETS TRIPLES - {final_bert}")
    print(f"\nMEAN ROUGE1 F1-SCORE BETWEEN NEWS AND TWEETS TRIPLES - {final_rouge1}")

    return final_evaluation, {"leven": final_leven, "FastText": final_fasttext, "BERT": final_bert,
                              "ROUGE1": final_rouge1}


if __name__ == '__main__':
    tweets_drs_dir = os.path.join(drs_files_path, "tweets")
    news_drs_dir = os.path.join(drs_files_path, "news")

    results_list = []
    for filename in os.listdir(news_drs_dir):
        topic_name = str(filename.split("_")[0])
        # if topic_name != "9c2abfde-02c5-4342-8a24-7c929090a6f8":
        #     continue
        print(topic_name)
        tweet_file = os.path.join(tweets_drs_dir, filename)
        if not os.path.isfile(tweet_file):
            print(f"No tweet file -> {tweet_file}")
            continue
        news_file = os.path.join(news_drs_dir, filename)
        evaluation_df, metrics = main(tweet_file, news_file, verbose=True)
        metrics["topic"] = topic_name

        results_list.append(metrics)
        evaluation_df.to_csv(os.path.join(results_path, topic_name+".csv"), index=False)

    with open(os.path.join(results_path, "metrics_by_topic.json"), "w+", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)
