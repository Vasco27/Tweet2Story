# Essentials
import pandas as pd


def tweets_per_topic(data):
    tweets_per_topic_count = data[["tweetID", "topic"]].groupby(["topic"]).count()
    tweets_per_topic_count.columns = ["topic_tweets"]
    tweets_per_topic_count.reset_index(inplace=True)

    tweet_per_topic_mean = tweets_per_topic_count["topic_tweets"].mean().round(2)

    return tweets_per_topic_count, tweet_per_topic_mean


def calculate_time_interval(date_end, date_init):
    if (not isinstance(date_end, pd.Series)) | (not isinstance(date_init, pd.Series)):
        raise TypeError(f"Parameter date_end and date_init must be of type {type(pd.Series())}. "
                        f"Instead they were of type {type(date_end)} and {type(date_init)}")

    time_interval = (date_end - date_init).astype("timedelta64[D]")
    return time_interval
