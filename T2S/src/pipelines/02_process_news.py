# Essential
import pandas as pd
from pathlib import Path
import os

root_dir = str(Path(__file__).parent.parent.parent.absolute())
datafile = os.path.join(root_dir, "datafiles", "signal1M_tweetir_data.csv")
TWEETS_DIR = os.path.join(root_dir, "resultfiles", "tweets_final")
NEWS_DIR = os.path.join(root_dir, "resultfiles", "news_final")


if __name__ == '__main__':
    tweetir_data = pd.read_csv(datafile)
    print(f"shape - {tweetir_data.shape[0]}")

    available_topics = []
    for filename in os.listdir(TWEETS_DIR):
        if filename.endswith(".txt"):
            available_topics.append(filename[:-4])

    temp = tweetir_data[tweetir_data["topic"].isin(available_topics)]
    news = temp.groupby("topic").last().reset_index()
    print(news.shape)

    for row in news.itertuples():
        with open(os.path.join(NEWS_DIR, str(row.topic)+".txt"), "w", encoding="utf-8") as f:
            f.write(row._11)

    print(f"News exported to - {NEWS_DIR}")
