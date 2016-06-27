import tweepy
import time
import cPickle as pickle
import csv, codecs, cStringIO
auth = tweepy.OAuthHandler("3TwNhhLMPHD2ao7eRzCn9z3tL", "yeiNTXCjB7h2BTsF6Jzh9fhOoF6MGrWFrOqxgP6G5Mbz8YJ0i8")

auth.set_access_token("1715782370-8QminAyiS5eGemy3HO9YyxiV8VRFpxFrxyvEIwi", "6Rrr2az4CIYfCkRwCVy603ET1s05GMOKEOINKakjCVdSX")

api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


query = '#apple -Filter:links -RT'
max_tweets = 10000
csvfile= open('apple_sentiments.csv', 'w')
wr = UnicodeWriter(csvfile)
count =0
for status in tweepy.Cursor(api.search, q=query,lang='en',count = 10000).items(max_tweets):
    wr.writerow([status.text])
    count +=1
csvfile.close()

print count



