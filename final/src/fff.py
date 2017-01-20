from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from collections import defaultdict
import csv
import re
import sys

df = open('./data/test.csv')

def clear(context):
    letters = re.sub("[^a-zA-Z]", " ", context)
    context = letters.lower().split()
    stopword = set(stopwords.words('english'))
    clear = [c for c in context if c not in stopword]
    return clear

def remove_html(context):
    cleaner = re.compile('<.*?>')
    clean_text = re.sub(cleaner,'',context)
    return clean_text

def pick_frequent(context):
    freq = FreqDist(context)
    return freq

Extra_wordpool = ['p','via','two','make','e','c','using','r','three', 'mu', 'eta', 'must', 'r', 'm', 'v']

reader = csv.DictReader(df)
output = open(sys.argv[1],'w')
writer=csv.writer(output)
writer.writerow(['id','tags'])

for idx,row in enumerate(reader):
    title = clear(row['title'])
    content = remove_html(row['content'])
    content = clear(content)
    ft = pick_frequent(title)
    fc = pick_frequent(content)
    common = set(fc).intersection(ft)
    temp = []
    if len(common) == 0:
        for t in title:
            if t not in Extra_wordpool:
                temp.append(t)
        writer.writerow([row['id'],' '.join(temp)])
    else:
        writer.writerow([row['id'],' '.join(common)])
