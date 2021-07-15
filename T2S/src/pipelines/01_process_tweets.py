# Essential
import pandas as pd
import pathlib
import re
import string
import time

from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
from flair.embeddings import WordEmbeddings

root_dir = str(pathlib.Path(__file__).parent.absolute())
datafile = "../../datafiles/signal1M_tweetir_data.csv"


def punctuate_sent(sent):
    if sent[-1] in ["?", "!", "."]:
        return sent

    return sent + "."


def remove_irrelevant_topics(data, tweets_threshold=3):
    topic_tweet_count = data[["topic", "topics.title"]].groupby("topic").count().reset_index()
    topic_tweet_count.rename(columns={"topics.title": "tweets_count"}, inplace=True)
    # Só apanho notícias com mais de 3 tweets associados
    relevant_topics = topic_tweet_count.loc[topic_tweet_count["tweets_count"] > tweets_threshold, "topic"]
    data = data[data["topic"].isin(relevant_topics)]
    print(f"total tweets - {data.shape[0]}")
    print(f"total topics - {data['topic'].unique().shape[0]}")

    return data


if __name__ == '__main__':
    tweetir_data = pd.read_csv(datafile)
    print(f"shape - {tweetir_data.shape[0]}")

    print("\nRemoving irrelevant tweets...")
    relevant_tweetir = tweetir_data[tweetir_data["relevancy"] == 2]
    print(f"shape - {relevant_tweetir.shape[0]}")

    print("\nRemoving irrelevant topics...\n"
          "Less than 3 tweets associated to the topic")
    relevant_tweetir = remove_irrelevant_topics(relevant_tweetir, tweets_threshold=3)

    # PROCESS TEXT #

    # Remove emojis - from: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    print("\nRemoving emojis")
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    relevant_tweetir["tweets.full_text"] = relevant_tweetir.apply(lambda x: emoji_pattern.sub(r'', x["tweets.full_text"]), axis=1)

    # Remove junk from tweets
    print("\nRemoving emails, urls, break lines, RT mentions and user mentions from tweet text...")
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r'[\n]+', '  ', x.strip() + ".  "))

    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r'[“\r\t\"\']|http\S+|www\.\S+|\S+@\S+\.(com|uk|co)|\S+\.com\S+|RT @\S+:|RT ', '', x)
    )

    # Remove hashtags at the end of the sentence
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.strip()
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r'( #\S+)*$', '', x)
    )
    # Remove user mentions at the end of the sentence
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r'( @\S+)*$', '', x)
    )

    # Remove everything before ":"
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r'^(.+?):', '', x)
    )

    # Remove # and @ from the middle of the text
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.replace("#", "")
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.replace("@", "")
    # relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.replace("&amp;", "&")

    # Remove non final punctuation from sentences
    remove = string.punctuation
    remove = remove.replace("?.!", "")
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
        lambda x: re.sub(r"[{}]".format(remove), ' ', x)
    )
    # Remove empty tweets
    relevant_tweetir = relevant_tweetir[~(relevant_tweetir["tweets.full_text"].str.strip() == "")]

    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(lambda x: " ".join(x.split()))
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.strip()

    # Keep tweets with more than 3 words
    relevant_tweetir = relevant_tweetir[relevant_tweetir["tweets.full_text"].str.split().apply(lambda x: len(x) > 3)]
    # Add period to the end of sentences
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(punctuate_sent)
    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].str.lower()

    print("\nRemove repeated retweets... >80% fast text similarity")
    fasttext_embeddings = WordEmbeddings('en-crawl')
    fasttext = Embeddings(fasttext_embeddings, min_similarity=0, model_id="FastText")
    model = PolyFuzz(fasttext)

    start = time.time()
    indexes_to_remove = []
    for topic in relevant_tweetir["topic"].unique():
        topic_tweets = relevant_tweetir.loc[relevant_tweetir["topic"] == topic, "tweets.full_text"]
        for index, tweet in topic_tweets.items():
            indexes = topic_tweets.index[topic_tweets.index != index]
            for ind in indexes:
                model.match(tweet.split(), topic_tweets.loc[ind].split())
                mean_sim = round(model.get_matches()["Similarity"].mean(), 2)
                if mean_sim > 0.8:
                    indexes_to_remove.append(ind)
                    break
    relevant_tweetir = relevant_tweetir[~relevant_tweetir.index.isin(indexes_to_remove)]
    end = time.time()

    print(indexes_to_remove)
    print(f"Computation time - {round(end - start, 2)} seconds")

    relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].drop_duplicates()
    relevant_tweetir.dropna(subset=["tweets.full_text"], inplace=True)
    relevant_tweetir = remove_irrelevant_topics(relevant_tweetir, tweets_threshold=3)

    print("\nExporting tweets to 'resultfiles/tweets_final/'")
    for topic in relevant_tweetir["topic"].unique():
        tweets = relevant_tweetir[relevant_tweetir["topic"] == topic]
        try:
            with open("../../resultfiles/tweets_final/"+str(topic)+".txt", "w", encoding="utf-8") as f:
                f.write('\n'.join(tweets["tweets.full_text"]))
        except UnicodeEncodeError as ex:
            print(repr(ex))
