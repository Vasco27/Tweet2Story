import pathlib


def get_root_path():
    return str(pathlib.Path(__file__).parent.absolute()) + "/../../"


def get_paths():
    """
    Based on the absolute path, makes the paths for the support directories,
    such as the datafiles directory and the resultfiles directory.
    :return: absolute path to datafiles and to export resultfiles
    """
    root_dir = get_root_path()
    data_dir = root_dir + "datafiles/"
    results_dir = root_dir + "resultfiles/"
    data_file = "signal1M_tweetir_processed.csv"

    path_to_data = data_dir + data_file
    print(f"Path to datafiles file - {path_to_data}")
    print(f"Path to output resultfiles directory - {results_dir}")

    return path_to_data, results_dir


def topic_tweets_documents(data, concat_sep="\n"):
    """
    Concatenate tweets from a news article into a single document.
    Make a list with the concatenated tweets from each article.
    :param concat_sep: String separator to concatenate tweets. Defaults to '\n'
    :param data: Pandas DataFrame with news articles and the respective tweets
    :return: List of concatenated tweets for each article
    Note: Expects to receive datafiles in signal1M format
    """
    tweets_texts = []
    for t in data["topic"].unique():
        t_data = data[data["topic"] == t]
        tweet_list = t_data["tweets.full_text"].tolist()
        concat_tweets = concat_sep.join(tweet_list)

        tweets_texts.append(concat_tweets)

    return tweets_texts
