"""
 1. split files
 2. clean
 3. convert POSIX to time of day by subtracting POSIX time of start of day
 3. word2vec style encoding of internet domains
 3. geocoding
 4. internet coding: find recurrent substrings, histogram of triples
 5. one hot encoding
 6. once we have a vector encoding, train regressor and train model that remembers user ids

 transactionRevenue
"""

import time
import pandas as pd
import json
from glob import glob
import numpy as np
from geocoding import geocode

DEVICE_BROWSERS = np.array(
    ['Amazon Silk', 'Android Browser', 'Android Webview', 'BlackBerry', 'Chrome', 'Coc Coc', 'Edge', 'Firefox',
     'Internet Explorer', 'Iron', 'LYF_LS_4002_12', 'MRCHROME', 'Maxthon', 'Mozilla', 'Mozilla Compatible Agent',
     'Nintendo Browser', 'Nokia Browser', 'Opera', 'Opera Mini', 'Puffin', 'Safari', 'Safari (in-app)', 'SeaMonkey',
     'Seznam', 'UC Browser', 'YaBrowser']
    , dtype=str)

DEVICE_OS = np.array(
    ['(not set)', 'Android', 'BlackBerry', 'Chrome OS', 'Firefox OS', 'Linux', 'Macintosh', 'Nintendo Wii',
     'Nintendo WiiU', 'Samsung', 'Windows', 'Windows Phone', 'Xbox', 'iOS']
    , dtype=str)

DEVICE_CATEGORIES = np.array(
    ['desktop', 'mobile', 'tablet']
    , dtype=str)

CHANNEL_GROUPING = np.array(
    ['(Other)', 'Affiliates', 'Direct', 'Display', 'Organic Search', 'Paid Search', 'Referral', 'Social']
    , dtype=str)

MON = np.arange(1, 13)
MDAY = np.arange(1, 32)
WDAY = np.arange(0, 7)
HOUR = np.arange(0, 24)

GEOCODER = json.load(open("geocoding_embedding.json", "r"))


def parse_date(s):
    time_object = time.strptime(s, '%Y%m%d')
    return time_object


def convert_posix(posix_time, day):
    return int(posix_time) - time.mktime(day)


def one_hot_encoding(value, all_values):
    res = np.zeros(len(all_values) + 1, dtype='float32')
    if value not in all_values:
        res[-1] = 1
    else:
        res[:-1] = all_values == value
    return 2 * (res - 0.5)


def encode(data):
    """
channelGrouping,date,device,fullVisitorId,
geoNetwork,sessionId,socialEngagementType,totals,
trafficSource,visitId,visitNumber,visitStartTime

    """
    visitor_id = data['fullVisitorId']
    date = parse_date(data['date'])

    totals = json.loads(data['totals'])
    transaction = 0
    if 'transactionRevenue' in totals:
        transaction = float(totals['transactionRevenue']) / 1e6
    if 'pageviews' not in totals:
        totals['pageviews'] = 0

    device = json.loads(data['device'])

    # TODO: center and rescale each part independently

    features = [
        # Time features
        *one_hot_encoding(date.tm_mon, MON),
        *one_hot_encoding(date.tm_mday, MDAY),
        *one_hot_encoding(date.tm_wday, WDAY),
        *one_hot_encoding(int(convert_posix(data['visitStartTime'], date) / (24 * 60 * 60)), HOUR),

        # totals features
        int(totals['visits']), int(totals['hits']), int(totals['pageviews']),

        # device features
        *one_hot_encoding(device['browser'], DEVICE_BROWSERS),
        *one_hot_encoding(device['operatingSystem'], DEVICE_OS),
        *one_hot_encoding(device['deviceCategory'], DEVICE_CATEGORIES),

        # channel grouping
        *one_hot_encoding(data['channelGrouping'], CHANNEL_GROUPING),

        # visit number
        data['visitNumber'],

        # Geo features
        *GEOCODER[geocode(data['geoNetwork'])]

        # Traffic source
    ]

    features = np.array(features, dtype='float32')

    return visitor_id, transaction, features


def encode_all_users(filename, user_features={}, user_scores={}):
    data = pd.read_csv(filename, dtype={'fullVisitorId': str, 'date': str})
    for i in range(len(data)):
        visitor_id, transaction, features = encode(data.loc[i])
        if visitor_id not in user_features:
            user_features[visitor_id] = []
            user_scores[visitor_id] = []
        user_features[visitor_id].append(features)
        user_scores[visitor_id].append(transaction)

    return user_features, user_scores


def split_users(user_features, user_scores, max_visits=10):
    users = list(user_scores.keys())
    for u in users:
        if len(user_scores[u]) > max_visits:
            n = len(user_scores[u])
            for i in range(n // max_visits):
                if max_visits * (i + 1) < n:
                    user_features[u + "+%s" % i] = user_features[u][max_visits * i:max_visits * (i + 1)]
                    user_scores[u + "+%s" % i] = user_scores[u][max_visits * i:max_visits * (i + 1)]
                else:
                    user_features[u + "+%s" % i] = user_features[u][-max_visits:]
                    user_scores[u + "+%s" % i] = user_scores[u][-max_visits:]
            del user_features[u]
            del user_scores[u]
    return user_features, user_scores


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(indent=4)

    user_features, user_scores = encode_all_users("train/20160801.csv")
    pp.pprint(user_scores)
