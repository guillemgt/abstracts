'''
    This file processes the data from the arXiv dataset on Kaggle to contain only the relevant information (i.e. categories, title and abstract) of math articles.
    The dataset must be placed in this folder, with the name 'dataset.json', and will output a file named 'prepared.json'
'''

import ijson
import re
import json

pattern = re.compile("(?<![A-Za-z])cs\.")
# For cs articles, use
# pattern = re.compile("(?<![A-Za-z])cs\.")

with open("prepared.json", "w") as of:
    with open("dataset.json", "r") as f:
        for line in f:
            objects = ijson.items(line, '')
            for obj in objects:
                if pattern.match(obj["categories"]):
                    of.write("{")
                    of.write('"categories": ')
                    of.write(json.dumps(obj["categories"]))
                    of.write(', "title": ')
                    of.write(json.dumps(re.sub(" +", " ", obj["title"].replace("\n", "").strip())))
                    of.write(', "abstract": ')
                    of.write(json.dumps(re.sub(" +", " ", re.sub(r'\n +', "\n", re.sub(r'\n([^ ])', r' \g<1>', obj["abstract"]))).strip()))
                    of.write("}\n")

        f.close()