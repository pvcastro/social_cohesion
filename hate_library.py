import pandas as pd
from url_analysis.url_library import extract_domain_and_suffix, extract_domain_name
from tqdm import tqdm
import numpy as np
import csv
from itertools import repeat
import time
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


home_dir = "C:/Users/16475/PycharmProjects/nigeria/"
data_dir = home_dir + "data/NG_communities_2/"


def load_ethnic_list():
    ethnic_list = [
        'christian',
        'muslim',
        'northern',
        'southern',
        'hausa',
        'fulani',
        'yoruba',
        'igbo',
        'women',
        'biafra',
        'arewa',
        'lgbt',
        'herdsmen',
        'eastern'
    ]
    return ethnic_list


def load_regex_ethnic_list():
    ethnic_list = [
        'christian|christians',
        'muslim|muslims|islam|islamic',
        'northern|northerner|northerners|arewa',
        'southern|southerner|southerners',
        'hausa|hausas',
        'fulani|fulanis',
        'yoruba|yorubas',
        'igbo|ibo|ibos|igbos',
        'women|woman|girl|girls|female|females',
        'lgbt|lgbtq|lgbtq+|gay|gays|lesbian|lesbians|transgender|transgenders',
        'herdsmen|herdsman',
        'eastern|easterner|easterners|biafra'
    ]
    return ethnic_list

def create_hate_tuples():
    ethnic_list = load_ethnic_list()
    toxic_list = [
        'stupid',
        'scum',
        'idiot',
        'cockroach',
        'parasite',
        'dog',
        'useless',
        'bastard',
        'moron',
        'dumb',
        'disgusting',
        'shit',
        'pig',
        'liar',
        'fool',
        'animal',
        'almajiri',
        'ewu',
        'mumu',
        # 'anuofia',
        'efulefu',
        'onye nzuzu',
        # 'oponu',
        'agba iya',
        'ode',
        'obun',
        # 'oloshi',
        'olodo',
        'didirin',
        'apoda',
        # 'arindin',
        'ole',
        'yeye',
        'slut',
        'bitch',
        'whore',
        'rape',
    ]

    hate_list = []
    # create list of tuples from two lists

    for i in range(len(ethnic_list)):
        temp_list = list(zip(repeat(ethnic_list[i]), toxic_list))
        hate_list = hate_list + temp_list
    return hate_list


def load_hate_words_xlsx_as_single_list():
    excel_words = os.path.join(home_dir, 'data/hate_speech/communities_and_hate_words.xlsx')
    df_excel = pd.read_excel(excel_words, sheet_name=['community', 'hate', 'combination of both'])
    y = df_excel['hate']
    z = df_excel['combination of both']
    # get round 2 hate keywords
    z = z['combination'].dropna().to_frame()
    y = y.loc[:, ['hate_keywords']]
    hate_list = list(y['hate_keywords'].append(z['combination']))
    return hate_list


def load_hate_words_xlsx_as_two_lists():
    excel_words = os.path.join(home_dir, 'data/hate_speech/communities_and_hate_words.xlsx')
    df_excel = pd.read_excel(excel_words, sheet_name=['community', 'hate', 'combination of both'])
    y = df_excel['hate']
    z = df_excel['combination of both']
    # get round 2 hate keywords
    combo = z['combination'].dropna().to_frame()
    y = y.loc[:, ['hate_keywords']]
    combined_words = list(combo['combination'])
    hate_list = list(y['hate_keywords'])

    targets = z['target'].dropna().to_frame()
    target_list = list(targets['target'])

    combined_dict = {}
    for i in range(0, len(target_list)):
        key = combined_words[i]
        value = target_list[i]
        combined_dict[key] = value

    return hate_list, combined_words, combined_dict

def load_hate_words_xlsx_as_two_lists_plural():
    excel_words = os.path.join(home_dir, 'data/hate_speech/communities_and_hate_words_plural.xlsx')
    df_excel = pd.read_excel(excel_words, sheet_name=['community', 'hate', 'combination of both'])
    y = df_excel['hate']
    z = df_excel['combination of both']
    # get round 2 hate keywords
    combo = z['combination'].dropna().to_frame()
    y = y.loc[:, ['hate_keywords']]
    combined_words = list(combo['combination'])
    hate_list = list(y['hate_keywords'])

    targets = z['target'].dropna().to_frame()
    target_list = list(targets['target'])

    combined_dict = {}
    for i in range(0, len(target_list)):
        key = combined_words[i]
        value = target_list[i]
        combined_dict[key] = value

    return hate_list, combined_words, combined_dict


def load_parquet_for_sample(filenames, data_dir):
    i = 0
    files = []
    for i in range(len(filenames)):
        df = pd.read_parquet(data_dir + filenames[i], engine='fastparquet')
        files.append(df)

    df = pd.concat(files, axis=0, ignore_index=True)
    return df


def popular(df):
    """ Take a df of users and split users by popularity given the top 90% RTS of users
        returns a df_popular and a df_not_popular"""
    data_dir_user = home_dir+'data/network/popularity/influencers_in_top_90.csv'

    df_users = pd.read_csv(data_dir_user)

    df_users['popular'] = 1
    df_users['user_source'] = df_users['user_source'].astype('str')

    df = df.merge(df_users, left_on="user_id", right_on="user_source", how='left', indicator=True)
    df['_merge'].value_counts()
    df['popular'] = df['popular'].fillna(0)
    # find subsample
    df_popular = df.loc[df['popular'] == 1, :]
    df_not_popular = df.loc[df['popular'] == 0, :]

    return df_popular, df_not_popular


def make_sample_from_two_lists(select_tweets, df_popular, df_not_popular):
    start = time.time()
    hate_list, combined_words = load_hate_words_xlsx_as_two_lists()

    print(f"Starting to look for word pairs in popular and not-popular tweets...")
    for i in tqdm(range(len(hate_list)), position=0, leave=False):
        df_sample_popular = df_popular.loc[np.logical_and(df_popular['text'].str.contains(fr'\b{hate_list[i][0]}\b', regex=True, case=False) == True,
                                                          df_popular['text'].str.contains(fr'\b{hate_list[i][1]}\b', regex=True, case=False) == True), :]
        df_sample_popular['ethnic_keyword'] = hate_list[i][0]
        df_sample_popular['hate_keyword'] = hate_list[i][1]
        # print(f"{df_sample_popular.shape[0]}")
        if df_sample_popular.shape[0] > 0:
            if df_sample_popular.shape[0] >= 5:
                df_pop = df_sample_popular.sample(n=5)
                # df_pop['popular'] = 1
                select_tweets = select_tweets.append(df_pop)
            else:
                df_pop = df_sample_popular
                # df_pop['popular'] = 1
                select_tweets = select_tweets.append(df_pop)
        else:
            pass
        df_sample_not_popular = df_not_popular.loc[
                                np.logical_and(
                                    df_not_popular['text'].str.contains(fr'\b{hate_list[i][0]}\b', regex=True,
                                                                        case=False) == True,
                                    df_not_popular['text'].str.contains(fr'\b{hate_list[i][1]}\b', regex=True,
                                                                        case=False) == True), :]
        df_sample_not_popular['ethnic_keyword'] = hate_list[i][0]
        df_sample_not_popular['hate_keyword'] = hate_list[i][1]
        if df_sample_not_popular.shape[0] > 0:
            if df_sample_not_popular.shape[0] >= 5:
                df_not_pop = df_sample_not_popular.sample(n=5)
                # df_not_pop['popular'] = 0
                select_tweets = select_tweets.append(df_not_pop)
            else:
                df_not_pop = df_sample_not_popular
                # df_not_pop['popular'] = 0
                select_tweets = select_tweets.append(df_not_pop)
        else:
            pass

    print(f"Starting to look for combined words ...")
    for i in tqdm(range(len(combined_words)), position=0, leave=False):
        df_sample_popular = df_popular.loc[
                                df_popular['text'].str.contains(fr'\b{combined_words[i]}\b', regex=True,
                                                                    case=False) == True, :]
        df_sample_not_popular = df_not_popular.loc[df_not_popular['text'].str.contains(fr'\b{combined_words[i]}\b', regex=True,
                                                                        case=False) == True, :]

        df_sample_popular['ethnic_keyword'] = combined_words[i]
        df_sample_popular['hate_keyword'] = combined_words[i]
        if df_sample_popular.shape[0] > 0:
            if df_sample_popular.shape[0] >= 5:
                df_pop = df_sample_popular.sample(n=5)
                # df_pop['popular'] = 1
                select_tweets = select_tweets.append(df_pop)
            else:
                df_pop = df_sample_popular
                # df_pop['popular'] = 1
                select_tweets = select_tweets.append(df_pop)
        else:
            pass

        df_sample_not_popular['ethnic_keyword'] = combined_words[i]
        df_sample_not_popular['hate_keyword'] = combined_words[i]
        if df_sample_not_popular.shape[0] > 0:
            if df_sample_not_popular.shape[0] >= 5:
                df_not_pop = df_sample_not_popular.sample(n=5)
                # df_not_pop['popular'] = 0
                select_tweets = select_tweets.append(df_not_pop)
            else:
                df_not_pop = df_sample_not_popular
                # df_not_pop['popular'] = 0
                select_tweets = select_tweets.append(df_not_pop)
        else:
            pass

    select_tweets = select_tweets.drop_duplicates('tweet_id')
    print(f"Sampling round {i} took" + "{:.2f} seconds to complete".format(time.time() - start))

    return select_tweets


def split_5_percent_files():
    filenames = [f"part-0000{i}-3eefb4c5-d676-4a11-acb5-33c407fe40e9-c000.snappy.parquet" for i in range(0, 10)] + \
                [f"part-000{i}-3eefb4c5-d676-4a11-acb5-33c407fe40e9-c000.snappy.parquet" for i in range(10, 100)] + \
                [f"part-00{i}-3eefb4c5-d676-4a11-acb5-33c407fe40e9-c000.snappy.parquet" for i in range(100, 1000)]
    split_filenames = np.array_split(filenames, 5)
    return split_filenames


def split_community_files(n):
    filenames = [f"part-0000{i}-362cb8bd-e764-46f5-9a21-33ee89333c64-c000.snappy.parquet" for i in range(0, 10)] + \
                [f"part-000{i}-362cb8bd-e764-46f5-9a21-33ee89333c64-c000.snappy.parquet" for i in range(10, 100)] + \
                [f"part-00{i}-362cb8bd-e764-46f5-9a21-33ee89333c64-c000.snappy.parquet" for i in range(100, 1000)]
    split_filenames = np.array_split(filenames, n)
    return split_filenames


def split_extended_community_files(n):
    filenames = [f"part-0000{i}-8403b41e-9bf2-419b-ba6f-d74b6b763be4-c000.snappy.parquet" for i in range(0, 10)] + \
                [f"part-000{i}-8403b41e-9bf2-419b-ba6f-d74b6b763be4-c000.snappy.parquet" for i in range(10, 100)] + \
                [f"part-00{i}-8403b41e-9bf2-419b-ba6f-d74b6b763be4-c000.snappy.parquet" for i in range(100, 1000)]
    split_filenames = np.array_split(filenames, n)
    return split_filenames


def run_for_split_5_percent(split_filenames, output_dir):
    start_2 = time.time()
    select_tweets = pd.DataFrame()
    i = 0
    for i in range(1, len(split_filenames)):
        print(f"starting split {i} ...")
        # one split at a time
        df = load_parquet_for_sample(split_filenames[i])
        df_popular, df_not_popular = popular(df)
        # empty dataframe to collect tweets
        temp_select_tweets = pd.DataFrame()
        temp_select_tweets = make_sample_from_two_lists(temp_select_tweets, df_popular, df_not_popular)

        # add class label column
        temp_select_tweets['class'] = None
        temp_select_tweets['index'] = temp_select_tweets.index

        temp_select_tweets = temp_select_tweets.loc[:, ['index', 'class', 'text', 'ethnic_keyword', 'hate_keyword', 'tweet_id', 'user_id']]
        temp_select_tweets['user_id'] = temp_select_tweets['user_id'].astype(str)
        temp_select_tweets['tweet_id'] = temp_select_tweets['tweet_id'].astype(str)
        print(f"saving split {i}, there are {temp_select_tweets.shape[0]} tweets")
        temp_select_tweets.to_csv(output_dir + "select_tweets_" + f"{i}" +".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
        select_tweets = select_tweets.append(temp_select_tweets, sort=True)

    print('Splits took {:.2f} seconds to complete'.format(time.time() - start_2))
    # temp_select_tweets = pd.read_csv(output_dir+"select_tweets_0.csv")
    # select_tweets = select_tweets.append(temp_select_tweets, sort=True)

    filenames_2 = ["select_tweets_0.csv",
                   "select_tweets_1.csv",
                   "select_tweets_2.csv",
                   "select_tweets_3.csv",
                   "select_tweets_4.csv",
                   ]

    select_tweets = pd.DataFrame()

    for i in range(len(filenames_2)):
        temp_df = pd.read_csv(output_dir + filenames_2[i])
        select_tweets = select_tweets.append(temp_df)

    print(f"saving all tweets, there are {select_tweets.shape[0]} tweets")
    select_tweets.to_csv(output_dir + "select_tweets_all.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    split_for_labeling_samples(select_tweets, output_dir, filename="tweets_2")


def separate_multiple_community_tweets(df):
    ethnic_list = load_ethnic_list()
    i = 0
    for i in tqdm(range(df.shape[0]), position=0, leave=True):
        count = 0
        for col in ethnic_list:
            if df.loc[i, col] == 1:
                count += 1
        df.loc[i, "count"] = count
    print(df['count'].value_counts())
    df.dropna(subset=['text'], inplace=True)
    # separate out tweets that mention 2 or more communities
    x = df.loc[df['count'] >= 2, :]
    # remove x from select tweets
    df = df.loc[~df['tweet_id'].isin(x['tweet_id']), :]
    return x, df


def split_for_labeling_samples(df, output_dir, filename):
    print(f"preping for labeling...")
    ethnic_list = load_ethnic_list()
    df['class'] = None
    df['index'] = df.index
    col_list = ['text', 'ethnic_keyword', 'hate_keyword', 'tweet_id', 'user_id'] + ethnic_list

    col_list = ['index', 'class'] + col_list
    df = df.loc[:, col_list]
    df.reset_index(inplace=True, drop=True)

    print(f"splitting for 6 random samples... ")
    shuffled = df.sample(frac=1)
    result = np.array_split(shuffled, 6)

    result[0]['name'] = 'Haaya'
    result[1]['name'] = 'Manu'
    result[2]['name'] = 'Pedro'
    result[3]['name'] = 'Luis'
    result[4]['name'] = 'Nausheen'
    result[5]['name'] = 'Niyati'

    print(f" Haaya has {result[0].shape[0]} tweets")
    print(f" Manuel has {result[1].shape[0]} tweets")
    print(f" Pedro has {result[2].shape[0]} tweets")
    print(f" Luis has {result[3].shape[0]} tweets")
    print(f" Niyati has {result[4].shape[0]} tweets")
    print(f" Nausheen has {result[5].shape[0]} tweets")


    print(f"saving randomized samples...")
    result[0].to_csv(output_dir + "haaya_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    result[1].to_csv(output_dir + "manu_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    result[2].to_csv(output_dir + "pedro_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    result[3].to_csv(output_dir + "luis_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    result[4].to_csv(output_dir + "nausheen_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    result[5].to_csv(output_dir + "niyati_"+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    print(f"done.")


def create_neg_sample_from_5_percent():
    "returns sample df with neg sample tweets"
    ethnic_list = load_ethnic_list()
    split_filenames = split_5_percent_files()

    df_com = pd.DataFrame()
    df_sample = pd.DataFrame()
    i=0
    # separate out tweets that mention at least 1 community
    for i in tqdm(range(len(split_filenames))):
        df = load_parquet_for_sample(split_filenames[i])
        # set indicator
        for word in ethnic_list:
            df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
            df.reset_index(inplace=True, drop=True)
        # keep tweets with at least 1 community
        for word in ethnic_list:
            temp_df = df.loc[df[word]==True, :]
            df_com = df_com.append(temp_df)
    # make sample
    for word in ethnic_list:
        temp_df = df_com.loc[df_com[word]==True, :]
        temp_df = temp_df.sample(n=50)
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)

    return df_sample


def create_neg_sample_from_community(n):
    "returns sample df with neg sample tweets"
    ethnic_list = load_ethnic_list()
    split_filenames = split_community_files(25)

    df_com = pd.DataFrame()
    df_sample = pd.DataFrame()
    i=0
    # separate out tweets that mention at least 1 community
    for i in tqdm(range(len(split_filenames))):
        df = load_parquet_for_sample(split_filenames[i], data_dir)
        # set indicator
        for word in ethnic_list:
            df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
            df.reset_index(inplace=True, drop=True)
        # keep tweets with at least 1 community
        for word in ethnic_list:
            temp_df = df.loc[df[word]==True, :]
            df_com = df_com.append(temp_df)
    # make sample
    for word in ethnic_list:
        temp_df = df_com.loc[df_com[word]==True, :]
        temp_df = temp_df.sample(n)
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)

    return df_sample


def create_neg_sample_from_extended_community(n, output_dir):
    "returns sample df with neg sample tweets"
    ethnic_list = load_ethnic_list()
    split_filenames = split_extended_community_files(50)
    split_filenames = split_filenames[0:25]

    df_com = pd.DataFrame()
    df_sample = pd.DataFrame()
    i=0
    # separate out tweets that mention at least 1 community
    for i in tqdm(range(len(split_filenames))):
        df = load_parquet_for_sample(split_filenames[i], data_dir)
        # set indicator
        for word in ethnic_list:
            df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
            df.reset_index(inplace=True, drop=True)
        # keep tweets with at least 1 community
        for word in ethnic_list:
            temp_df = df.loc[df[word]==True, :]
            df_com = df_com.append(temp_df)
    # make sample
    for word in ethnic_list:
        temp_df = df_com.loc[df_com[word]==True, :]
        temp_df = temp_df.sample(n)
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)
    df_sample.to_csv(output_dir+"neg_sample.csv")
    return df_sample


def calculate_binary_values_and_sample(df):
    hate_list = load_hate_words_xlsx_as_single_list()
    ethnic_list = load_ethnic_list()
    hate_tweets = pd.DataFrame()
    df_neg_sample = ()
    dict = {True: 1, False: 0}
    print(f"ethnic list and neg samples...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
        temp_df = df.loc[df[word] == 1, :]
        temp_df = temp_df.sample(n=1)
        temp_df['ethnic_keyword'] = word
        df_neg_sample = df_neg_sample.append(temp_df)
    df.reset_index(inplace=True, drop=True)

    print(f"hate list and hate tweets...")
    for word in tqdm(hate_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
        temp_df = df.loc[df[word] == 1, :]
        temp_df['hate_keyword'] = word
        hate_tweets = hate_tweets.append(temp_df)
    df.reset_index(inplace=True, drop=True)
    hate_tweets['tweet_id'].drop_duplicates(inplace=True)

    # calculate product columns
    i =0
    new_col_names = []
    print(f" calculating word pairs ...")
    for i in tqdm(range(len(ethnic_list)), position=0, leave=True):
        for word in hate_list:
            new_col_name = ethnic_list[i] + '_' + word
            new_col_names.append(new_col_name)
            df[new_col_name] = df[ethnic_list[i]] * df[word]
    columns = new_col_names
    return df, columns, hate_tweets, df_neg_sample


def calculate_binary_values(df):
    hate_list = load_hate_words_xlsx_as_single_list()
    ethnic_list = load_ethnic_list()
    dict = {True: 1, False: 0}
    df['hate_keyword'] = None
    df['ethnic_keyword'] = None
    print(f"checking tweets for community words...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
    df.reset_index(inplace=True, drop=True)
    # set ethnic column
    df_sample = pd.DataFrame()
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = df.loc[df[word] == 1, :]
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)
    df = df_sample
    print(f"checking tweets for hate words...")
    for word in tqdm(hate_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: True if x.lower().find(word)!=-1 else False)
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
    df.reset_index(inplace=True, drop=True)
    df.drop_duplicates('tweet_id', inplace=True)

    # calculate product columns
    i =0
    new_col_names = []
    print(f" calculating word pairs ...")
    for i in tqdm(range(len(ethnic_list)), position=0, leave=True):
        for word in hate_list:
            new_col_name = ethnic_list[i] + '_' + word
            new_col_names.append(new_col_name)
            df[new_col_name] = df[ethnic_list[i]] * df[word]
    columns = new_col_names
    return df, columns


def sample_hate_tweets_by_community(n, hate_tweets, output_dir, filename):
    df_sample = pd.DataFrame()
    ethnic_list = load_ethnic_list()
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = hate_tweets.loc[hate_tweets[word]==1, :]
        temp_df = temp_df.sample(n)
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)
        cols = ['tweet_id', 'user_id', 'text'] + ethnic_list
        hate_tweets = hate_tweets.loc[:, cols]
    hate_tweets.to_csv(output_dir+"sample_3500_hate_tweets.csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)


def sample_hate_tweets_by_community_and_popularity(n, df, output_dir, filename):
    df_sample = pd.DataFrame()
    ethnic_list = load_ethnic_list()
    hate_list = load_hate_words_xlsx_as_single_list()
    hate_df = pd.DataFrame()
    # set hate keyword
    print(f"setting hate keyword...")
    for word in tqdm(hate_list, position=0, leave=True):
        temp_df = df.loc[df[word]==1, :]
        temp_df['hate_keyword'] = word
        hate_df = hate_df.append(temp_df)
    # sample and set ethnic keyword
    print(f"setting ethnic keyword and sampling...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = hate_df.loc[hate_df[word]==1, :]
        temp_df = temp_df.sample(n)
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)
    df_sample.to_csv(output_dir+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    return df_sample


def sample_hate_tweets_and_combined_by_community_and_popularity(n_1, n_2, df, output_dir, filename):
    df_sample = pd.DataFrame()
    ethnic_list = load_ethnic_list()
    hate_list, combined_words, combined_dict = load_hate_words_xlsx_as_two_lists()
    hate_df = pd.DataFrame()
    # set hate keyword
    print(f"setting hate keyword for {df.shape[0]} tweets...")
    for word in tqdm(hate_list, position=0, leave=True):
        temp_df = df.loc[df[word]==1, :]
        temp_df['hate_keyword'] = word
        hate_df = hate_df.append(temp_df)
    # sample and set ethnic keyword
    print(f"setting ethnic keyword and sampling {n_1} tweets for {len(ethnic_list)} communities...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = hate_df.loc[hate_df[word]==1, :]
        if temp_df.shape[0] >= n_1:
            temp_df = temp_df.sample(n_1)
            temp_df['ethnic_keyword'] = word
            df_sample = df_sample.append(temp_df)
        else:
            temp_df['ethnic_keyword'] = word
            df_sample = df_sample.append(temp_df)
    print(f"setting hate word, ethnic word and sampling {n_2} tweets for the {len(combined_words)} combined words...")
    for word in tqdm(combined_words, position=0, leave=True):
        temp_df = df.loc[df[word]==1, :]
        if temp_df.shape[0] >= n_2:
            temp_df = temp_df.sample(n_2)
            temp_df['hate_keyword'] = word
            temp_df['ethnic_keyword'] = combined_dict[word]
            df_sample = df_sample.append(temp_df)
        else:
            temp_df['hate_keyword'] = word
            temp_df['ethnic_keyword'] = combined_dict[word]
            df_sample = df_sample.append(temp_df)
    print(f"saving {df_sample.shape[0]} sample tweets")
    df_sample.to_csv(output_dir+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    return df_sample


def balanced_sample_hate_tweets_and_combined_by_community(n_1, n_2, df, output_dir, filename):
    df_sample = pd.DataFrame()
    ethnic_list = load_ethnic_list()
    hate_list, combined_words, combined_dict = load_hate_words_xlsx_as_two_lists()
    hate_df = pd.DataFrame()
    # set hate keyword
    print(f"setting hate keyword for {df.shape[0]} tweets...")
    for word in tqdm(hate_list, position=0, leave=True):
        temp_df = df.loc[df[word]==1, :]
        temp_df['hate_keyword'] = word
        hate_df = hate_df.append(temp_df)
    # sample and set ethnic keyword
    print(f"setting ethnic keyword and sampling {n_1} tweets for {len(ethnic_list)} communities...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = hate_df.loc[hate_df[word]==1, :]
        if temp_df.shape[0] >= 1:
            for hate_word in tqdm(hate_list, position=0, leave=True):
                temp_df2 = temp_df.loc[temp_df[hate_word]==1, :]
                if temp_df2.shape[0] >= n_1:
                    temp_df2 = temp_df2.sample(n_1)
                    temp_df2['ethnic_keyword'] = word
                    df_sample = df_sample.append(temp_df2)
                elif temp_df2.shape[0] >= 1:
                    temp_df2['ethnic_keyword'] = word
                    df_sample = df_sample.append(temp_df2)
                else:
                    pass
        else:
            pass
    print(f"setting hate word, ethnic word and sampling {n_2} tweets for the {len(combined_words)} combined words...")
    for word in tqdm(combined_words, position=0, leave=True):
        temp_df = df.loc[df[word]==1, :]
        if temp_df.shape[0] >= n_2:
            temp_df = temp_df.sample(n_2)
            temp_df['hate_keyword'] = word
            temp_df['ethnic_keyword'] = combined_dict[word]
            df_sample = df_sample.append(temp_df)
        else:
            temp_df['hate_keyword'] = word
            temp_df['ethnic_keyword'] = combined_dict[word]
            df_sample = df_sample.append(temp_df)
    print(f"saving {df_sample.shape[0]} sample tweets")
    df_sample.to_csv(output_dir+filename+".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)
    return df_sample


def create_community_sample(split_filenames, output_dir):
    print(f"starting sampling...")
    for i in tqdm(range(len(split_filenames)), position=0, leave=True):
        print(f" ")
        print(f"loading file {i+1} out of {len(split_filenames)} ")
        df = load_parquet_for_sample(split_filenames[i], data_dir)
        print(f"calculating values..")
        print(f"creating neg sample and finding hate tweets...")
        df, columns, hate_tweets, df_neg_sample = calculate_binary_values_and_sample(df)
        # collect 5 x 10 com random tweets/ split
        # collect all hate tweets
        print(f"found {hate_tweets.shape[0]} hate tweets")
        print(f"saving {hate_tweets.shape[0]} hate tweets")
        hate_tweets.to_csv(output_dir + "hate_tweets" + f"_{i}" + ".csv", index=False, encoding='utf-8-sig',
                       quoting=csv.QUOTE_NONNUMERIC)
        print(f"saving {df_neg_sample.shape[0]} random tweets")
        df_neg_sample.to_csv(output_dir + "neg_sample" + f"_{i}" + ".csv", index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)


def match_regex(regex, tweet: str):
    return len([match for match in regex.finditer(tweet.lower())]) > 0


def match_ethnic_insult(tweet: str, ethnicities_regex, insults_regex, debug: bool = False):

    if match_regex(ethnicities_regex, tweet) and match_regex(insults_regex, tweet):
        if debug:
            print(tweet)
        return True
    return False


def match_combined_insult(tweet: str, combined_regex, debug: bool = False):

    if match_regex(combined_regex, tweet):
        if debug:
            print(tweet)
        return True
    return False


def create_community_regex_sample(split_filenames, ethnicities_regex, insults_regex, combined_regex, output_dir, filename):
    start = time.time()

    print(f"starting sampling...")
    i = 0
    for i in tqdm(range(len(split_filenames)), position=0, leave=True):
        regex_matched_df = pd.DataFrame()
        print(f" ")
        print(f"loading file {i+1} out of {len(split_filenames)} ")
        df = load_parquet_for_sample(split_filenames[i], data_dir)
        print(f" ")
        print(f"calculating regex for {df.shape[0]} tweets...")
        regex_matched = []
        for idx, row in tqdm(df.iterrows(), position=0, leave=True):
            if match_ethnic_insult(row['text'], ethnicities_regex, insults_regex, debug=False):
                regex_matched.append(row)
        #         print(f'Tweet {idx}: {row["text"]}')
            if match_combined_insult(row['text'], combined_regex, debug=False):
                # print(f'Tweet {idx}: {row["text"]}')
                regex_matched.append(row)
        #
        temp_df = pd.DataFrame(regex_matched)
        regex_matched_df = regex_matched_df.append(temp_df)
        print(f" ")
        print(f"found {regex_matched_df.shape[0]} tweets, saving to disk...")
        regex_matched_df.to_csv(output_dir+filename+f"_{i}.csv")
        print(f" ")
    print("Regex took {:.2f} seconds to complete".format(time.time() - start))


def load_regex_hate_tweets(output_dir, num_files):
    hate_files = [f"regex_matched_tweets_{i}.csv" for i in range(0, num_files)]
    hate_tweets = pd.DataFrame()
    for i in range(len(hate_files)):
        temp_df = pd.read_csv(output_dir+hate_files[i])
        hate_tweets = hate_tweets.append(temp_df)
    return hate_tweets


def load_regex():
    hate_list, combined_words, combined_dict = load_hate_words_xlsx_as_two_lists_plural()
    ethnic_list = load_regex_ethnic_list()
    joined_ethnicities = '|'.join([ethnicity.strip() for ethnicity in ethnic_list])
    joined_insults = '|'.join([insult.strip() for insult in hate_list])
    joined_combined_words = '|'.join([word.strip() for word in combined_words])

    ethnicities_regex = re.compile(fr"\b(?:{joined_ethnicities})\b", re.IGNORECASE)
    insults_regex = re.compile(fr"\b(?:{joined_insults})\b", re.IGNORECASE)
    combined_regex = re.compile(fr"\b(?:{joined_combined_words})\b", re.IGNORECASE)
    return ethnicities_regex, insults_regex, combined_regex, combined_dict


def get_keywords_regexes(keyword_list):
    return {keyword: re.compile(fr"\b(?:{keyword})\b", re.IGNORECASE) for keyword in keyword_list}


def match_regex_map(word: str, regexes_map, tweet: str):
    return word in regexes_map and match_regex(regex=regexes_map[word], tweet=tweet)


def match_regex_word(word, ethnicities_regexes, insults_regexes, tweet: str):
    return match_regex_map(word=word, regexes_map=ethnicities_regexes, tweet=tweet) or \
           match_regex_map(word=word, regexes_map=insults_regexes, tweet=tweet)


def load_regex_for_sampling():
    hate_list, combined_words, combined_dict = load_hate_words_xlsx_as_two_lists_plural()
    ethnic_list = load_regex_ethnic_list()
    ethnicities_regexes = get_keywords_regexes(ethnic_list)
    insults_regexes = get_keywords_regexes(hate_list)
    return ethnicities_regexes, insults_regexes


def calculate_binary_values_with_regex(df):
    hate_list, combined_words, combined_dict = load_hate_words_xlsx_as_two_lists_plural()
    hate_list = hate_list + combined_words
    ethnic_list = load_regex_ethnic_list()
    ethnicities_regexes, insults_regexes = load_regex_for_sampling()
    dict = {True: 1, False: 0}
    df['hate_keyword'] = None
    df['ethnic_keyword'] = None
    print(f"checking tweets for community words...")
    for word in tqdm(ethnic_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: match_regex_word(word, ethnicities_regexes, insults_regexes, x))
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
    df.reset_index(inplace=True, drop=True)
    # set ethnic column
    df_sample = pd.DataFrame()
    for word in tqdm(ethnic_list, position=0, leave=True):
        temp_df = df.loc[df[word] == 1, :]
        temp_df['ethnic_keyword'] = word
        df_sample = df_sample.append(temp_df)
    df = df_sample
    print(f"checking tweets for hate words...")
    for word in tqdm(hate_list, position=0, leave=True):
        df[word] = df['text'].apply(lambda x: match_regex_word(word, ethnicities_regexes, insults_regexes, x))
        # convert True to 1 and False to 0
        df = df.replace({word: dict})
    df.reset_index(inplace=True, drop=True)
    df.drop_duplicates('tweet_id', inplace=True)

    # calculate product columns
    i =0
    new_col_names = []
    print(f" calculating word pairs ...")
    for i in tqdm(range(len(ethnic_list)), position=0, leave=True):
        for word in hate_list:
            new_col_name = ethnic_list[i] + '_' + word
            new_col_names.append(new_col_name)
            df[new_col_name] = df[ethnic_list[i]] * df[word]
    columns = new_col_names
    return df, columns


def heatmap(df, output_path, hate_list, ethnic_list):
    data_rows = []
    for word in hate_list:
        temp_df = df.loc[df['hate_keyword'] == word, :]
        data_row = []
        for ethnicity in ethnic_list:
            temp_df2 = temp_df.loc[temp_df['ethnic_keyword'] == ethnicity, :]
            num_rows = temp_df2.shape[0]
            data_row.append(num_rows)
        data_rows.append(data_row)

    data_rows_shares = [[round(100.0 * val / float(sum(data_row)), 1) if sum(data_row)>0 else 0 for val in data_row] for data_row in data_rows]
    bad_rows = []
    for i in range(len(hate_list)):
        if sum(data_rows[i])==0:
            bad_rows.append(i)
    for index in sorted(bad_rows, reverse=True):
        del hate_list[index]
        del data_rows[index]
    sns.set(rc={'figure.figsize': (10, 83)})
    sns.heatmap(data_rows_shares, cmap="YlGnBu",
                cbar=False,
                yticklabels=hate_list,
                xticklabels=ethnic_list, annot=True)  # cmap="YlGnBu",, annot = True
    plt.yticks(rotation=45)
    plt.xlabel("community")
    plt.ylabel("hate keyword")
    # plt.savefig(output_path)
    plt.show()



