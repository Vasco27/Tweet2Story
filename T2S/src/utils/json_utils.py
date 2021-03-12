# Essentials
import pandas as pd

# import stats_utils
from T2S.src.utils import stats_utils

from _ctypes import PyObj_FromPtr
import json
import re


# Both of these JSON encoder classes taken from
# https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr


def convert_to_datetime(data, col):
    data[col] = pd.to_datetime(data[col])
    return data


def tweets_dates_per_topic(data):
    topic_first_tweet = data[["topic", "tweets.created_at"]].groupby(["topic"]).min()
    topic_last_tweet = data[["topic", "tweets.created_at"]].groupby(["topic"]).max()
    topic_published_date = data[["topic", "topics.published"]].groupby(["topic"]).max()

    topic_tweet_time_interval = stats_utils.calculate_time_interval(topic_last_tweet["tweets.created_at"],
                                                                    topic_first_tweet["tweets.created_at"])
    topic_tweet_time_interval.name = "tweets.time_interval"

    topic_first_tweet["tweets.created_at"] = topic_first_tweet["tweets.created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    topic_last_tweet["tweets.created_at"] = topic_last_tweet["tweets.created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    topic_published_date["topics.published"] = topic_published_date["topics.published"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return topic_first_tweet, topic_last_tweet, topic_published_date, topic_tweet_time_interval
