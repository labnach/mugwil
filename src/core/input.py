"""
This module provides a method to get ABC tunes from different sources, basic ABC pre-processing and functions to
generate the dataset and feed the model.

The main function of this module is `get_dataset`. Go to its docstring for details.
"""
import re
from collections import Counter

import numpy as np
import requests


def get_abcs(source):
    """Returns a list of strings, each string is an ABC tune.

    `source` will be resolved in this order:
        - First try to read files from source as a directory.
        - If it's not a dir (but it is a pathlib.Path) try to read text.
        - If not a Path, try to GET text from source as URL (or list of URLs).
        - If them all fail, print the exception and return None.
    """
    try:
        files = list(source.iterdir())
        abcs = [f.read_text().strip() for f in files]
    except NotADirectoryError:  # is pathlib.Path, but not a dir.
        abcs = source.read_text()
    except AttributeError:  # not a pathlib.Path. read from list of URLs.
        if type(source) is str:
            source = [source]
        try:
            abcs = ''
            for url in source:
                print(f'{__file__}: downloading {url}')
                abcs += requests.get(url).text
        except requests.exceptions.MissingSchema as e:  # bad URL format.
            abcs = None
            print(e)
    except FileNotFoundError as e:
        abcs = None
        print(e)
    finally:
        # if it's an ABC tunebook, split it.
        if type(abcs) is str:
            s = re.split(r'X:\s*\d+\n', abcs)
            abcs = ['X: 1\n' + abc.strip() for abc in s if abc.strip() != '']
    return abcs


def remove_abc_comments(abc_string):
    lines = []
    for line in abc_string.split('\n'):
        if line.startswith('%%MIDI program'):
            lines.append(line.strip())
        else:
            lines.append(line.split('%')[0].strip())
    return '\n'.join([l for l in lines if l != ''])


def get_dataset(abcs, remove_comments=True, normalize_x=False):
    """Returns a list of tuples (encoded_abc_text, labels_for_each_timestep)
    for each element in `abcs`, and a token-to-index dict.
    `abcs` must be a list of strings, each string representing an ABC tune..

    Note for remove_comments: There's a special type of comment,
        "%%MIDI program ##". Depending on the program number, the resulting
        sound changes when converted to MIDI. If there's no such comment it
        sounds like "%%MIDI program 0" (piano).
    """
    if remove_comments:
        abcs = [remove_abc_comments(abc) for abc in abcs]

    # generate token to integer mapping.
    text = ''.join(abcs)
    counter = sorted(Counter(text).most_common())
    token2id = {t: i for i, (t, _) in enumerate(counter)}
    num_unique_chars = len(token2id)

    # encode text and generate labels.
    dataset = []
    for abc in abcs:
        x = np.asarray([token2id[c] for c in abc], dtype=np.int32)
        y = np.zeros(shape=(len(abc), num_unique_chars), dtype=np.int32)
        # set one-hot labels with fancy indexing.
        y[np.arange(len(abc)), np.roll(x, shift=-1)] = 1
        if normalize_x:
            n = len(token2id)/2
            x = np.vectorize(lambda i: (i - n) / n)(x)
        dataset.append((x, y))

    return dataset, token2id
