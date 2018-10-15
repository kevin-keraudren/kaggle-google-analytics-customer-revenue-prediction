import pandas as pd
import json
from glob import glob

TO_CLEAN = ['device', 'trafficSource']

# voluntarily do not clean 'geoNetwork'

NOT_AVAILABLE = "not available in demo dataset"


def clean(f_in, f_out):
    data = pd.read_csv(f_in)

    for i in range(len(data)):
        for field in TO_CLEAN:
            d = json.loads(data.loc[i, (field)])
            keys = list(d.keys())
            for k in keys:
                if d[k] == NOT_AVAILABLE:
                    del d[k]
            if field == 'trafficSource':
                del d['adwordsClickInfo']
            data.loc[i, (field)] = json.dumps(d)

    data.to_csv(f_out, index=False)


# for f in glob("train/*.csv"):
#     clean(f, f)

for f in glob("test/*.csv"):
    clean(f, f)

# clean("train/20160801.csv", "20160801_clean.csv")
