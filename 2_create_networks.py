#coding:utf-8
import os
import re
import base64
import quopri
import json
import math
import networkx as nx
import pandas as pd 
from tqdm import tqdm
from collections import Counter

'''
# convert to UTF-8 coding
def text_encoding(encoded_words):
    # return ascii code
    if '=' not in encoded_words:
        return encoded_words
    try:
        encoded_word_regex = r'=\?{1}(.+)\?{1}([B|Q])\?{1}(.+)\?{1}='
        charset, encoding, encoded_text = re.match(encoded_word_regex, encoded_words).groups()
        if encoding == 'B':
            byte_string = base64.b64decode(encoded_text)
        elif encoding == 'Q':
            byte_string = quopri.decodestring(encoded_text)
        return byte_string.decode(charset)
    except:
        return encoded_words
'''

# processing technical nets
print('processing technical nets...')

c_path = './monthly_data/commits/'
projects = os.listdir(c_path)
to_path = './network_data/commits/'
if not os.path.exists(to_path):
    os.makedirs(to_path)

for project in tqdm(projects):
    technical_net = {}
    project_name, period = project.replace('.csv', '').split('__')
    df = pd.read_csv(c_path+project)
    df.query('is_bot == False and is_coding == True', inplace=True)
    df = df[df['dealised_author_full_name'].notna()]
    for index, row in df.iterrows():
        file_path = row['file_name']
        # file extension = '.' + file_path.split('/')[-1].split('.')[-1].split(' ')[0]
        file_name = file_path.split('/')[-1]
        author_name = row['dealised_author_full_name']
        if file_name not in technical_net:
            technical_net[file_name] = {}
        if author_name not in technical_net[file_name]:
            technical_net[file_name][author_name] = {}
            technical_net[file_name][author_name]['weight'] = 0
        technical_net[file_name][author_name]['weight'] += 1

    #save as directed graph
    g = nx.DiGraph(technical_net)
    # add disconnected nodes
    g.add_nodes_from(technical_net.keys())
    nx.write_edgelist(g, to_path + '{}__{}.edgelist'.format(project_name, str(period)), delimiter='##', data=["weight"])


# ---------------- processing social nets ----------------------
print('processing social nets...')
to_path = './network_data/emails/'
if not os.path.exists(to_path):
    os.makedirs(to_path)
e_path = './monthly_data/emails/'
projects = os.listdir(e_path)

# get sender-receiver timestamp
sender_dic = {'project': [], 'message_id':[], 'sender':[], 'receiver':[], \
              'timestamp':[], 'broadcast':[]}

for project in tqdm(projects):

    social_net = {}
    emailID_to_author = {}
    project_name, period = project.replace('.csv', '').split('__')
    df = pd.read_csv(e_path+project)
    df.query('is_bot == False', inplace=True)
    df = df[df['dealised_author_full_name'].notna()]
    
    for index, row in df.iterrows():
        message_id = row['message_id'].strip()
        # print(row['dealised_author_full_name'])
        sender_name = row['dealised_author_full_name']
        timestamp = row['date']
        emailID_to_author[message_id] = (sender_name, timestamp)

    # raise KeyError
    for index, row in df.iterrows():
        message_id = row['message_id']
        references = row['references']
        sender_name = row['dealised_author_full_name']
        timestamp = row['date']

        # ignores if this email does not reply to previous emails
        if pd.isna(references) or references == 'None':
            continue

        references = [r.strip() for r in references.replace('\n', ' ').replace('\t', ' ').split(' ') if r.strip()]

        # deal with the issue that a line breaker exists in message_id:
        # e.g., <4\n829AB62.6000302@apache.org>
        new_refs = set()
        for i in range(len(references)-1):
            if '<' in references[i] and '>' not in references[i] and '<' not in references[i+1] and '>' in references[i+1]:
                new_refs.add(references[i] + references[i+1])
        for r in references:
            if '<' in r and '>' in r:
                new_refs.add(r)

        references = new_refs
        for reference_id in references:
            if reference_id not in emailID_to_author:
                continue
            prev_author, prev_timestamp = emailID_to_author[reference_id]
            # if it's the same person, continue
            if prev_author == sender_name:
                continue

            # add to the sender-receiver mapping
            # sender drops an email to previous author
            sender_dic['project'].append(project_name)
            sender_dic['message_id'].append(message_id)
            sender_dic['sender'].append(sender_name)
            sender_dic['receiver'].append(prev_author)
            sender_dic['timestamp'].append(timestamp)
            sender_dic['broadcast'].append(0)    

            # since the previous author sent information to 
            # this receiver since the receivier replied.
            sender_dic['project'].append(project_name)
            sender_dic['message_id'].append(reference_id)
            sender_dic['sender'].append(prev_author)
            sender_dic['receiver'].append(sender_name)
            sender_dic['timestamp'].append(prev_timestamp)
            sender_dic['broadcast'].append(1)

            if sender_name not in social_net:
                social_net[sender_name] = {}
            if prev_author not in social_net:
                social_net[prev_author] = {}

            # if node B replies node A, it means B sends signal to A
            if prev_author not in social_net[sender_name]:
                social_net[sender_name][prev_author] = {}
                social_net[sender_name][prev_author]['weight'] = 0
            social_net[sender_name][prev_author]['weight'] += 1

            # if node B replies node A, it means A also sent signal to B
            if sender_name not in social_net[prev_author]:
                social_net[prev_author][sender_name] = {}
                social_net[prev_author][sender_name]['weight'] = 0
            social_net[prev_author][sender_name]['weight'] += 1

    #save as directed graph
    g = nx.DiGraph(social_net)
    # add disconnected nodes
    g.add_nodes_from(social_net.keys())
    nx.write_edgelist(g, to_path + '{}__{}.edgelist'.format(project_name, str(period)), delimiter='##', data=["weight"])

df = pd.DataFrame.from_dict(sender_dic)
df.to_csv('./email_mapping.csv', index=False)
print('all done!')

