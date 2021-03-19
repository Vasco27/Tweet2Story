import pathlib
import re


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


def flatten_list(list_to_flatten):
    """
    Simply flattens a list of nested lists.

    :param list_to_flatten: list of nested lists to be flattened
    :return: the flattened list
    """
    return [item for sublist in list_to_flatten for item in sublist]


def value_indexes_in_list(ref_list, value):
    """
    Finds the indexes where a value is contained in a list.

    :param ref_list: the list from where to search the value
    :param value: the value to search for within the list
    :return: a list of indexes corresponding to the queried value
    """
    return [idx for idx, val in enumerate(ref_list) if val == value]


def multiple_index_list(ref_list, indexes):
    """
    Index a list with multiple item indexes.
    Example: having indexes - [1, 2, 3] we want to access ref_list[1, 2, 3].

    :param ref_list: list to index
    :param indexes: indexes of values within the range of the list
    :return: the values in the indexes within the list
    """
    return [ref_list[i] for i in indexes]


def trim_whitespaces(text):
    """
    Trim whitespaces from an entire text or a list of words. Including whitespaces preceding punctuation.

    :param text: str or list of str to trim
    :return: str fully trimmed of whitespaces

    Reference:
    https://stackoverflow.com/questions/18878936/how-to-strip-whitespace-from-before-but-not-after-punctuation-in-python
    """
    # todo: Make util class to check for errors (function check_if_string_or_string_list, e.g.)
    if not isinstance(text, (str, list)):
        raise ValueError(f"Parameter text must be of type str or list. Instead it was {type(text)}")
    if isinstance(text, list):
        if len(text) <= 0:
            raise ValueError(f"List must contain at least one value.")
        if any(not isinstance(w, str) for w in text):
            raise ValueError(f"List must contain only str values.")

    if isinstance(text, list):
        text = ' '.join(text)

    temp_str = ' '.join(text.split())  # Trims whitespaces
    normalized_text = re.sub(r'\s([?.!",](?:\s|$))', r'\1', temp_str)

    return normalized_text
