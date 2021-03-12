# Essential
import pandas as pd
import pathlib
import re
import os
import shutil
import string
import warnings

# Preprocessing utils
import demoji
import T2S.src.utils.json_utils as json_utils
import T2S.src.utils.stats_utils as stat_utils
from functools import reduce

demoji.download_codes()

root_dir = str(pathlib.Path(__file__).parent.absolute())
print("Root directory:\n", root_dir)
data_dir = "../../datafiles/"
tweetir_data_file = "signal1M_tweetir_data.csv"
tweetir_json_file = "signal1M_tweetir_json.json"
data_path = root_dir + data_dir + tweetir_data_file

tweetir_data = pd.read_csv(data_path)
print(tweetir_data.shape[0])

# Only consider somewhat-relevant and completely-relevant tweets
print("\nRemoving irrelevant tweets...")
relevant_tweetir = tweetir_data[tweetir_data["relevancy"] > 0]
old_shape = relevant_tweetir.shape[0]
print(f"Lost a total of {tweetir_data.shape[0] - old_shape} tweets by removing irrelevant ones.")
print(f"Total of tweets - {old_shape}")

# Remove repeated retweets
print("\nRemove repeated retweets...")
relevant_tweetir = relevant_tweetir.drop_duplicates(subset=["tweets.full_text"], ignore_index=True)
new_shape = relevant_tweetir.shape[0]
print(f"Total of tweets - {new_shape}")
print(f"Lost a total of {old_shape - new_shape} tweets when removing duplicated retweets.")


# Remove emojis
def find_emojis(tweet_list):
    emoji_count = 0
    for tweet in tweet_list:
        ems = demoji.findall(tweet)
        if len(ems):
            print(ems)
            emoji_count += 1

    print(f"Found {emoji_count} tweets with emojis.")


print("\nRemoving emojis...")
find_emojis(relevant_tweetir["tweets.full_text"])
relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(lambda x: demoji.replace(x))
find_emojis(relevant_tweetir["tweets.full_text"])

# Remove junk from tweets
print("\nRemoving emails, urls, break lines, RT mentions and user mentions from tweet text...")
relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
    lambda x: re.sub(r'[\n]+', '  ', x.strip() + ".  "))

relevant_tweetir["tweets.full_text"] = relevant_tweetir["tweets.full_text"].apply(
    lambda x: re.sub(r'[“\r\t?\"\']|http\S+|www\.\S+|\S+@\S+\.(com|uk|co)|\S+\.com\S+|@\S+|RT.+: ', '', x))

# Remove junk from topics
print("\nRemoving emails, urls and break lines from topic content...")
relevant_tweetir["topics.content"] = relevant_tweetir["topics.content"].apply(
    lambda x: re.sub(r'[\n]+', '  ', x.strip() + ".  "))

relevant_tweetir["topics.content"] = relevant_tweetir["topics.content"].apply(
    lambda x: re.sub(r'[“\r\t?\"\']+|http\S+|www\.\S+|\S+@\S+\.(com|uk|co)\S+|\.com\S+', '', x))

# Export to csv
print("\nExporting to csv...")
relevant_tweetir.to_csv(root_dir + data_dir + "signal1M_tweetir_processed.csv", index=False)
print(f"Exported to {root_dir + data_dir + 'signal1M_tweetir_processed.csv'}")

# Export to JSON
print("\nExporting to JSON...")
relevant_tweetir.to_json(root_dir + data_dir + tweetir_json_file, orient="records", date_format="iso",
                         indent=4)

# Export to JSON as one directory per topic
# Convert dates to datetime objects
relevant_tweetir = json_utils.convert_to_datetime(relevant_tweetir, "tweets.created_at")
relevant_tweetir = json_utils.convert_to_datetime(relevant_tweetir, "topics.published")

# Group datafiles by topic for statistic purposes
# get nr of tweets per topic and the time interval of tweets about a certain topic
tweets_per_topic, _ = stat_utils.tweets_per_topic(relevant_tweetir)
topic_first_tweet, topic_last_tweet, topic_published_date, topic_tweet_time_interval = \
    json_utils.tweets_dates_per_topic(relevant_tweetir)
print(tweets_per_topic.shape[0], topic_first_tweet.shape[0], topic_last_tweet.shape[0], topic_published_date.shape[0])

# Merge all statistics into a single DataFrame
dfs = [tweets_per_topic, topic_first_tweet, topic_last_tweet, topic_tweet_time_interval, topic_published_date]
df_final = reduce(lambda left, right: pd.merge(left, right, on="topic", suffixes=("_initial", "_final")), dfs)
df_final.drop(columns="topics.published", inplace=True)
print(df_final.shape)

# Merge statistics with the complete processed DataFrame
data_to_json = pd.merge(relevant_tweetir, df_final, on="topic")

# Create tree where each directory represents a topic
json_dir_path = root_dir + data_dir + "tweetir_json/"
shutil.rmtree(json_dir_path)
os.mkdir(json_dir_path)

# Choose important columns for the datafiles
topics_data = data_to_json.iloc[:, [0, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
tweets_data = data_to_json.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]

# Fill tree directory with a folder for each news article
# With its topic information and its tweets information
chars = re.escape(string.punctuation+" ")
print(topics_data["topic"].unique().shape[0])
for topic in topics_data["topic"].unique():
    topic_data = topics_data[topics_data["topic"] == topic]

    topic_info = topic_data.iloc[0]
    title = topic_info["topics.title"].strip()
    title = re.sub(r"["+chars+"]", "_", title)
    topic_dir = json_dir_path + title + "/"
    os.mkdir(topic_dir)

    topic_file_name = "news_" + topic_info["topics.published"].strftime("%Y%m%d%H%M%S") + "_" + topic + ".json"
    try:
        topic_info.to_json(topic_dir + topic_file_name, date_format="iso", indent=4)
    except FileNotFoundError:
        warnings.warn(f"For some reason the path {topic_dir} was not found.")

    tweets_info = tweets_data[tweets_data["topic"] == topic]
    init_date_str = re.sub(r"[- :]", "", topic_info["tweets.created_at_initial"])
    final_date_str = re.sub(r"[- :]", "", topic_info["tweets.created_at_final"])
    tweets_file_name = "tweets_" + init_date_str + "_" + final_date_str + "_" + topic + ".json"
    try:
        tweets_info.to_json(topic_dir + tweets_file_name, date_format="iso", orient="records", indent=4)
    except FileNotFoundError:
        warnings.warn(f"For some reason the path {topic_dir} was not found.")

# Zip final directory
shutil.make_archive(json_dir_path[:-1], 'zip', json_dir_path)
