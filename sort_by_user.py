import pandas as pd
import json
from glob import glob
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

all_visitors = {}
all_scores = {}
all_transaction_count = {}

all_browsers = {}
all_os = {}
all_categories = {}

all_channel_groupings = {}

total_transaction = 0
total_visits = 0

for f in glob("train/*.csv"):
    data = pd.read_csv(f, dtype={'fullVisitorId': str})
    for i in range(len(data)):
        total_visits += 1
        visitor = data.loc[i, ('fullVisitorId')]
        if visitor not in all_visitors:
            all_visitors[visitor] = 1
            all_scores[visitor] = 0
            all_transaction_count[visitor] = 0
        else:
            all_visitors[visitor] += 1
        totals = json.loads(data.loc[i, ('totals')])
        if 'transactionRevenue' in totals:
            all_scores[visitor] += int(totals['transactionRevenue'])
            all_transaction_count[visitor] += 1
            total_transaction += 1

        device = json.loads(data.loc[i, ('device')])
        if device['browser'] not in all_browsers:
            all_browsers[device['browser']] = 1
        else:
            all_browsers[device['browser']] += 1
        if device['operatingSystem'] not in all_os:
            all_os[device['operatingSystem']] = 1
        else:
            all_os[device['operatingSystem']] += 1
        if device['deviceCategory'] not in all_categories:
            all_categories[device['deviceCategory']] = 1
        else:
            all_categories[device['deviceCategory']] += 1

        channel_grouping = data.loc[i, ('channelGrouping')]
        if channel_grouping not in all_channel_groupings:
            all_channel_groupings[channel_grouping] = 1
        else:
            all_channel_groupings[channel_grouping] += 1

        # if data.loc[i, ('socialEngagementType')] == "Not Socially Engaged":
        #     print(data.loc[i, ('socialEngagementType')])

# exit(0)
#
# for key, value in sorted(all_visitors.items(), key=lambda x: x[1]):
#     print("%s: %s -- %s (%s)" % (key, value, np.log(all_scores[key] + 1), all_transaction_count[key]))

# keep only items with more than 10 elements
for d in [all_browsers, all_os, all_categories, all_channel_groupings]:
    pp.pprint(d)
    keys = list(d.keys())
    keys = list(filter(lambda k: d[k] > 100, keys))
    print(list(sorted(keys)))
    print()

print("Transaction probability:", total_transaction / total_visits)
