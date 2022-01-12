import pandas as pd
from NLP.hate_sampling_scripts.hate_library import *
import re
import os

home_dir = "C:/Users/16475/PycharmProjects/nigeria/"
data_dir = home_dir + "data/NG_communities_2/"
# set output directory
output_dir = data_dir + 'round_2_regex/'
split_filenames = split_extended_community_files(50)
# create eval set by spliting the ~30M tweets in to two sets
split_filenames = split_filenames[0:25]
# load communities
ethnic_list = load_regex_ethnic_list()
# create regex for communities, hate words and combined words
ethnicities_regex, insults_regex, combined_regex, combined_tuples = load_regex()

# isolate hate tweets with regex matching
create_community_regex_sample(split_filenames, ethnicities_regex, insults_regex, combined_regex, output_dir, 'regex_matched_tweets')

# load negative sample
neg_sample = pd.read_csv(output_dir+"neg_sample.csv")
neg_sample, cols_neg = calculate_binary_values(neg_sample)
neg_sample.drop_duplicates('tweet_id', inplace=True)

# load hate tweets from disk
hate_tweets = load_regex_hate_tweets(output_dir, 25)
# calculate binary values for hate tweets with regex
hate_tweets, cols = calculate_binary_values_with_regex(hate_tweets)
# shuffle hate tweets & drop duplicate text
hate_tweets = hate_tweets.sample(frac=1)
hate_tweets.drop_duplicates('text', inplace=True)
hate_tweets['user_id'] = hate_tweets['user_id'].astype(str)

col_list = ['text', 'ethnic_keyword', 'hate_keyword', 'tweet_id', 'user_id'] + ethnic_list
neg_sample = neg_sample.loc[:, col_list]

# get 300 hate tweets per community stratified by word pair (3 each)
hate_tweets = balanced_sample_hate_tweets_and_combined_by_community(3, 5, hate_tweets, output_dir, 'hate_tweets_sample.csv')

# combine hate tweets sample with neg sample
sample_tweets = hate_tweets.append(neg_sample)
sample_tweets.drop_duplicates('text', inplace=True)
sample_tweets.drop_duplicates('tweet_id', inplace=True)

# save sample tweets
sample_tweets.to_csv(output_dir+"all_sample_tweets.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
sample_tweets = pd.read_csv(output_dir+"all_sample_tweets.csv")

# separate out tweets that mention 2+ communities (x) one-community tweets (y)
# adjust community cols
x, y = separate_multiple_community_tweets(sample_tweets)

# split one-community tweets 6 ways
split_for_labeling_samples(y, output_dir+'samples/', filename='tweets_round_2')
split_for_labeling_samples(x, output_dir+'samples/', filename='tweets_round_2_plus_com')

##########################################################################################
# create matrix
sample_tweets, cols = calculate_binary_values(sample_tweets)
sample_tweets.drop_duplicates('tweet_id', inplace=True)

matrix = sample_tweets.groupby(["ethnic_keyword", "hate_keyword"]).size().reset_index(name="sample_count")
matrix = matrix.pivot(index='hate_keyword', columns='ethnic_keyword', values='sample_count')
matrix['hate_keyword'] = matrix.index
cols_hate = ['hate_keyword'] + ethnic_list
matrix = matrix.loc[:, cols_hate]
matrix = matrix.sort_values(by=ethnic_list)

community_dist = sample_tweets['ethnic_keyword'].value_counts().to_frame()
community_dist['count'] = community_dist['ethnic_keyword']
community_dist['ethnic_keyword'] = community_dist.index

sample_tweets = x.append(y)
hate_dist = sample_tweets['hate_keyword'].value_counts().to_frame()
hate_dist['count'] = hate_dist['hate_keyword']
hate_dist['hate_keyword'] = hate_dist.index

matrix.to_csv(output_dir + "matrix.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
community_dist.to_csv(output_dir + "community_dist.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
hate_dist.to_csv(output_dir + "hate_dist.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)







