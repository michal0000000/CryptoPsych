from stream_v2 import IDPrinter
from stream_v2 import get_followers
import sys
import subprocess
import time

# implement pip as a subprocess:
"""
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--update', 
'tensorflow'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--update', 
'tensorflow-gpu'])
"""

consumer_key = "YOUR CONSUMER KEY"
consumer_secret_key = "YOUR CONSUMER SECRET"
ACESS_TOKEN = 'YOUR ACESS TOKEN'
ACESS_TOKEN_SECRET = 'YOUR ACESS TOKEN SECRET'


while True:
    printer = IDPrinter(consumer_key, consumer_secret_key, ACESS_TOKEN,
                    ACESS_TOKEN_SECRET)

    #printer.filter(follow=get_followers()) # FOLLW YOUR FOLLOWERS
    following = ["$sol",'#sol','#solana'] # REPLACE WITH YOUR LIST
    printer.set_following(following)
    printer.filter(track=following)
    printer.disconnect()


    print("Stream ended... Trying to reconnect in one minute")
    time.sleep(60)
