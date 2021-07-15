import pandas as pd
import os
import json
from importlib import import_module
from pathlib import Path
from openie import StanfordOpenIE

from nltk.tokenize import sent_tokenize

# Custom modules
from T2S.src.utils import drs_parser as parser
evaluation = import_module('03_drs2triples')

ROOT_DIR = os.path.join(Path(__file__).parent.parent.parent)
DRS_DIR = os.path.join(ROOT_DIR, "datafiles", "drs_files")
TWEETS_DIR = os.path.join(ROOT_DIR, "resultfiles", "tweets_final")
results_path = os.path.join(ROOT_DIR, "resultfiles", "triple_evaluation")

if __name__ == '__main__':
    with StanfordOpenIE() as client:
        news_drs_dir = os.path.join(DRS_DIR, "news")

        results_list = []
        for filename in os.listdir(news_drs_dir):
            tweet_name = filename.split("_")[0] + ".txt"
            tweet_file = os.path.join(TWEETS_DIR, tweet_name)
            if not os.path.isfile(tweet_file):
                print(f"No tweet file -> {tweet_file}")
                continue

            news_file = os.path.join(news_drs_dir, filename)

            with open(tweet_file, "r", encoding="utf-8") as f:
                tweets = sent_tokenize(''.join(f.readlines()))

            kg_list = []
            t_actors = {}
            actor_id = 1
            for sent in tweets:
                triples = client.annotate(sent)
                if len(triples) != 0:
                    triple = triples[0]

                    actor1_id = "T"+str(actor_id)
                    t_actors[actor1_id] = triple["subject"]
                    actor_id += 1

                    actor2_id = "T"+str(actor_id)
                    t_actors[actor2_id] = triple["object"]
                    actor_id += 1

                    kg_list.append({"edge1": actor1_id, "node": triple["relation"], "edge2": actor2_id})

            tweets_kg = pd.DataFrame(kg_list)
            print(tweets_kg)
            print(t_actors)
            news_actors, _, news_events = parser.get_graph_data(news_file)
            news_kg = pd.DataFrame(news_events, columns=["edge1", "node", "edge2"])
            news_actors = {key: val.lower() for key, val in news_actors.items()}

            try:
                evaluation_df, metrics = evaluation.main(news_actors, news_kg, t_actors, tweets_kg, verbose=True)
            except Exception as ex:
                print(repr(ex))
                continue
            metrics["topic"] = filename.split("_")[0]
            metrics["n_triples"] = tweets_kg.shape[0]
            results_list.append(metrics)

        with open(os.path.join(results_path, "stanford_evaluation.json"), "w+", encoding="utf-8") as f:
            json.dump(results_list, f, indent=4)
