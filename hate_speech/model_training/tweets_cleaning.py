#!/usr/bin/env python
# coding: utf-8

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List
from nltk.corpus import stopwords

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

tag_pattern = re.compile(r"\<\S+\>")


def match_tag(token: str) -> bool:
    return len(tag_pattern.findall(token)) > 0


def load_text_preprocessor() -> TextPreProcessor:
    return TextPreProcessor(
        # terms that will be normalized
        omit=['url', 'email', 'percent', 'money', 'phone', 'user',
              'time', 'url', 'date', 'number'],
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"elongated", "repeated"},
        #     annotate={"hashtag", "allcaps", "elongated", "repeated",
        #         'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter",

        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )


# In[5]:


common_abbreviations = ['imho', 'ayfkmwts', 'gtfooh', 'oh', 'rlrt', 'gmafb', 'nbd', 'smh', 'idk', 'stfu',
                        'nfw', 'irl', 'nsfw', 'sfw', 'fml', 'fwiw', 'qotd', 'lmao', 'hotd',
                        'ftw', 'btw', 'bfn', 'afaik', 'lol', 'ty', 'yw', 'fb', 'li', 'ff']

twitter_stopwords = [
                        'btw', 'ctp', 'dm', 'fb', 'fr', 'hr', 'ht', 'htt',
                        'hv', 'kp', 'kpk', 'kv', 'lg', 'mt', 'nd', 'np', 'ns',
                        'pdm', 'pk', 'pls', 'plz', 'ppl', 'pr', 'ps', 'psf', 'psl',
                        'rd', 'rn', 'rs', 'rss', 'rt', 'rts', 'sb', 'smh',
                        'tbh', 'th', 'thx', 'tl', 'tv', 'wtf', 'yr', 'yrs'
                    ] + common_abbreviations


def load_all_stopwords() -> List[str]:
    all_stopwords = set(ENGLISH_STOP_WORDS)
    all_stopwords.update(set(twitter_stopwords))
    all_stopwords.update(set(stopwords.words('english')))
    return list(all_stopwords)


all_stopwords = load_all_stopwords()

text_processor = load_text_preprocessor()


def ekphrasis_clean(sentence: str, remove_stopwords: bool) -> str:
    if remove_stopwords:
        return " ".join([token for token in text_processor.pre_process_doc(sentence) if
                         (not match_tag(token) and token not in all_stopwords)])
    else:
        return " ".join([token for token in text_processor.pre_process_doc(sentence) if not match_tag(token)])
