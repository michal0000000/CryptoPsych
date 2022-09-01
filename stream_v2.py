import mysql.connector
from datetime import datetime
import tweepy
from mysql.connector import errorcode
import time
from preprocess import preprocess_tweet
from tensorflow.keras.models import load_model
import numpy as np
import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_followers():
    consumer_key = "YOUR CONSUMER KEY"
    consumer_secret_key = "YOUR CONSUMER SECRET"
    ACESS_TOKEN = 'YOUR ACESS TOKEN'
    ACESS_TOKEN_SECRET = 'YOUR ACESS TOKEN SECRET'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
    auth.set_access_token(ACESS_TOKEN, ACESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    followers = api.get_friend_ids(stringify_ids=True)
    return followers

def get_feature_vector(tweet,vocab):
    """
    Generates a feature vector for each tweet where each word is
    represented by integer index based on rank in vocabulary.
    """
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


class IDPrinter(tweepy.Stream):

    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
      super().__init__(consumer_key, consumer_secret, access_token, access_token_secret)

      self.database_connection = mysql.connector.connect(
          host="HOST",
          user="USER",
          passwd="PASS",
          port="PORT",
          database="database",
          use_unicode=True,
          collation='collation' # DONT DELETE
      )

      self.FREQ_DIST_FILE = 'train-processed-freqdist.pkl'
      self.BI_FREQ_DIST_FILE = 'my_datasets/train-processed-freqdist-bi.pkl'
      self.TRAIN_PROCESSED_FILE = 'my_datasets/train-processed.csv'
      self.TEST_PROCESSED_FILE = 'my_datasets/test-processed.csv'
      self.GLOVE_FILE =  'glove-seeds-50d.txt'
      self.dim = 50

      self.vocab_size = 90000
      self.batch_size = 500
      self.max_length = 40
      self.filters = 600
      self.kernel_size = 3
      self.vocab = utils.top_n_words(self.FREQ_DIST_FILE, self.vocab_size, shift=1)

      np.random.seed(1337)
      
      self.dbcursor = self.database_connection.cursor()
      self.count = 1
      self.now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      self.following = []
      self.glove_vectors = self.get_glove_vectors(self.vocab)
      self.model = load_model('model/m2_e10.hdf5')

      print("Streming...")

    def set_following(self,following):
      self.following = following

    def on_limit(self, status):
        print("Rate Limit Exceeded, Sleep for 15 Mins")
        time.sleep(15 * 60)
        return True

    def on_status(self, status):

        print(status.id_str)

        # IF RETWEETED RETURN
        retweeted = False
        try:
            is_retweeted = status.retweeted_status
            #print("Skippin retweeted: " + str(status.id_str))
            retweeted = True
        except:
            pass

        end = False
        commit = False

        if status.lang == 'en' and retweeted == False:

            sql = "INSERT INTO polished_labeled_tweets (tweet_id,user,posted,related_to,tweet,clean_tweet,sentiment) VALUES (%s,%s,%s,%s,%s,%s,%s);"
      
            try:
                tweet_text = status.extended_tweet["full_text"]
            except:
                tweet_text = status.text

            try:

              print("TWEET TEXT: "   + tweet_text)

            except:
                pass

            clean_tweet = preprocess_tweet(tweet_text)

            # Create and embedding matrix
            embedding_matrix = np.random.randn(self.vocab_size + 1, self.dim) * 0.01
            # Seed it with GloVe vectors
            for word, i in self.vocab.items():
                glove_vector = self.glove_vectors.get(word)
                if glove_vector is not None: # remove words that dont have vector
                    embedding_matrix[i] = glove_vector

            test_tweets = [get_feature_vector(clean_tweet,self.vocab)]
            test_tweets = pad_sequences(test_tweets, maxlen=self.max_length, padding='post')
            predictions = self.model.predict(test_tweets, batch_size=128, verbose=1)

            sentiment = np.round(predictions[:, 0]).astype(int)

            print(status.id_str)
            print(sentiment[0])

            values = (
                status.id_str,
                status.user.id_str,
                status.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                str(self.following),
                tweet_text,
                clean_tweet,
                int(sentiment[0])
            )

            if self.database_connection.is_connected() == False:
                time.sleep(60 * 2)
                self.database_connection = mysql.connector.connect(
                    host="HOST",
                    user="USER",
                    passwd="PASS",
                    port="PORT",
                    database="database",
                    use_unicode=True,
                    collation='collation'  # DONT DELETE
                )

            try:
                self.dbcursor.execute(sql, values)
            except mysql.connector.Error as error:
                if error.errno == errorcode.ER_DUP_ENTRY:
                    pass
                else:
                    print(error)
                    commit = True

            self.count += 1

            if end == True:
                print("breaking")
                return

            if self.count % 20 == 0:
                commit = True
                print(f"{self.count} tweets mined so far - {self.now}")

            if commit == True:
                print("Commited.")

                try:
                    self.database_connection.commit()
                except:
                    print("Ran into a commit problem, restarting in one minute.")
                    time.sleep(60)
                    self.database_connection = mysql.connector.connect(
                        host="HOST",
                        user="USER",
                        passwd="PASS",
                        port="PORT",
                        database="database",
                        use_unicode=True,
                        collation='collation'  # DONT DELETE
                    )

                    try:
                        self.database_connection.commit()
                    except:
                        print("Cant commit for some reason. Line 212")

                commit = False

    def on_error(self, status_code):
        if status_code == 420:
            return False
        if status_code == 406:
            return False

    def following(self,following):
      self.following = following
      print(f"Following: {self.following}")
    
    def get_glove_vectors(self,vocab):
      """
      Extracts glove vectors from seed file only for words present in vocab.
      """
      print('Looking for GLOVE seeds')
      glove_vectors = {}
      found = 0
      with open(self.GLOVE_FILE, 'r',encoding='utf-8') as glove_file:
          for i, line in enumerate(glove_file):
              tokens = line.strip().split()
              word = tokens[0]
              if vocab.get(word):
                  vector = [float(e) for e in tokens[1:]]
                  glove_vectors[word] = np.array(vector)
                  found += 1
      return glove_vectors
    
    def process_tweets(self,raw_tweet,vocab, test_file=True):
      """
      Generates training X, y pairs.
      """
      tweets = []
      labels = []
      
      feature_vector = get_feature_vector(raw_tweet,vocab)
      tweets.append(feature_vector)
            
      return tweets, np.array(labels)

    
