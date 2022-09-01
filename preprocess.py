import re
import sys
from nltk.stem.porter import PorterStemmer
import csv
import pandas as pd

def handle_emojis(tweet):
    regrex_pattern = re.compile(pattern="["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F600-\U000E007F"
        "]+", flags=re.UNICODE)
    result = regrex_pattern.sub(r'', str(tweet))
    return result

def remove_urls(tweet,token=""):
    return re.sub("http[s]?:\/\/.[^\s]*", f"{token}", str(tweet))

def remove_mentions(tweet,token=""):
    return re.sub("\@([a-zA-Z0-9_]+)",f"{token}",tweet)

def remove_cashtags(tweet,token=""):
    return re.sub('\$([a-zA-Z]+)',f"{token}", tweet) # remove cashtags

def add_whitespace_between_punctuation(tweet):
    return re.sub(r"([,.!?])",r' \1 ', tweet)

def add_whitespace_between_numbers(tweet,token=""): # replace or remove numbers and add whitespace: 100X => <NUMBER> X
    return re.sub(r"\d+",f" {token} ",tweet)

def remove_excess_whitespace(tweet):
    return re.sub("\s+"," ",tweet)

def remove_special_chars(tweet):
    # does not remove: :   ,.!?
    tweet = tweet.replace("&amp;", " and ")
    tweet = tweet.replace("-", " ")
    tweet = tweet.replace("\n", " ")
    tweet = tweet.replace("/", " ")
    tweet = tweet.replace("&lt;", "<")
    tweet = tweet.replace("&gt;", ">")
    tweet = tweet.replace("%", " percent ")
    tweet = tweet.replace("https:", "")
    tweet = re.sub('[$=@_*#&‘\/{})\]:\|\%\;\<>\~\-\,\'\’\(\[+ˆ”"]', "", tweet)
    return tweet

def reduce_char_sequence(tweet):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", tweet)


def keep_hashtags_if_common_crypto_word(tweet):
    corpus = ['altcoin', 'altcoins', 'bitcoin', 'crypto',
              'cryptocurrency', 'btc', 'eth', 'binance',
              'nft', 'nfts', 'ethereum', 'bnb',
              'blockchain', 'defi', '100xgem', 'metaverse',
              'trading', 'altseason', 'cryptotrading',
              'web3', 'alts', 'dyor']

    hashtags = re.findall('\#([a-zA-Z0-9]+)', tweet)  # extract hashtags
    for i in range(len(hashtags)):
        hashtags[i] = hashtags[i].lower()

    for tag in hashtags:
        regex = r"#\b{}\b".format(tag)
        if tag in corpus:  # ak sa slovo nachadza v corpuse
            tweet = re.sub(regex, f"{tag}", tweet)  # ponechaj hashtag
        else:
            tweet = re.sub(regex, "", tweet) # vymaz hashtag
    return tweet


def elongate_common_crypto_abbervations(tweet):
    abbervations = {
        'dyor': "do your own research",
        'lfg': 'lets fucking go',
        'hodling': 'holding',
        'btd': 'buy the dip',
        'defi': 'decentralised finance',
        'fomo': 'fear of missing out',
        'fud': 'fear uncertainty and doubt',
        'roi': 'return on investment',
        'dao': 'decentralized autonomous organization',
        'daos': 'decentralized autonomous organizations',
        "p2p": 'peer to peer',
        'ta': 'technical analysis',
        'fa': 'fundamental analysis',
        'ytd': 'year to date'
    }
    abbers_list = list(abbervations.keys())
    tweet_words = tweet.split(" ")

    for i in range(len(tweet_words)):
        if tweet_words[i].lower() in abbers_list:
            tweet_words[i] = abbervations[tweet_words[i].lower()]

    return " ".join(tweet_words)

def preprocess_tweet(tweet):
    processed_tweet = []

    # Convert to lower case
    tweet = tweet.lower()
    
    # MY SETTINGS
    tweet = handle_emojis(tweet)
    tweet = remove_urls(tweet," link ") # Replaces URLs with the word <LINK>
    tweet = remove_mentions(tweet," mention ") # Replace @handle with the word <MENTION>
    tweet = remove_cashtags(tweet," coin ") # Replace $cashtag with <COIN>
    tweet = add_whitespace_between_punctuation(tweet)
    tweet = keep_hashtags_if_common_crypto_word(tweet)
    tweet = add_whitespace_between_numbers(tweet," number ")
    tweet = elongate_common_crypto_abbervations(tweet)
    tweet = reduce_char_sequence(tweet)
    tweet = remove_excess_whitespace(tweet)
    tweet = remove_special_chars(tweet)
    tweet = remove_excess_whitespace(tweet)
    tweet = tweet.lstrip()
    tweet = tweet.lstrip()

    return tweet

def get_df_from_csv_path(path,cols=['tweet_id','tweet','clean_tweet','cashtags','vader_sentiment']):
    tweets = pd.read_csv(path)
    tweets.columns = cols
    tweets = tweets.drop(['cashtags','vader_sentiment'],axis=1,errors='ignore')
    return tweets


def preprocess_csv(csv_file_name, processed_file_name, test_file=False):

    save_to_file = open(processed_file_name, 'w', encoding='utf8')
    count = 1
    with open(csv_file_name, 'r', encoding='utf8') as file:

        file = csv.reader(file, delimiter=',')

        for line in file:

            if count == 1:
                count += 1
                continue

            tweet_id = line[0]
            tweet = line[1]
            sentiment = line[2]
            
            processed_tweet = preprocess_tweet(tweet)
            if not test_file:
                save_to_file.write('%s,%s,%s\n' %
                                   (tweet_id, sentiment, processed_tweet))
            else:
                save_to_file.write('%s,%s\n' %
                                   (tweet_id, processed_tweet))
            count += 1
    save_to_file.close()
    print('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python preprocess.py <raw-CSV>')
        exit()
    use_stemmer = False
    csv_file_name = sys.argv[1]
    processed_file_name = sys.argv[1][:-4] + '-processed.csv'
    if use_stemmer:
        porter_stemmer = PorterStemmer()
        processed_file_name = sys.argv[1][:-4] + '-processed-stemmed.csv'
    print("working")
    preprocess_csv(csv_file_name, processed_file_name, test_file=False)
