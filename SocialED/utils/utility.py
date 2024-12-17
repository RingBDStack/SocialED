# -*- coding: utf-8 -*-
"""A set of utility functions to support outlier detection.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

import os
import torch
import shutil
import numbers
import requests
import warnings
import numpy as np
from importlib import import_module

from ..metrics import *

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min


def construct_graph_from_df(df, G=None):
    """Construct a graph from a DataFrame containing social media data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing social media data with columns:
        tweet_id, user_mentions, user_id, entities, sampled_words
    G : networkx.Graph, optional (default=None)
        Existing graph to add nodes/edges to. If None, creates new graph.
        
    Returns
    -------
    G : networkx.Graph
        Graph with nodes for tweets, users, entities and words, and edges between them.
    """
    import networkx as nx
    
    if G is None:
        G = nx.Graph()
        
    for _, row in df.iterrows():
        # Add tweet node
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True

        # Add user nodes
        user_ids = row['user_mentions']
        user_ids.append(row['user_id']) 
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        # Add entity nodes
        entities = row['entities']
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True

        # Add word nodes
        words = ['w_' + each for each in row['sampled_words']]
        G.add_nodes_from(words)
        for each in words:
            G.nodes[each]['word'] = True

        # Add edges between tweet and other nodes
        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities] 
        edges += [(tid, each) for each in words]
        G.add_edges_from(edges)

    return G

def tokenize_text(text, max_length=512):
    """Tokenize text for social event detection tasks.
    
    Parameters
    ----------
    text : str
        The input text to tokenize.
    max_length : int, optional (default=512)
        Maximum length of tokenized sequence.
        
    Returns
    -------
    tokens : list
        List of tokenized words/subwords.
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic tokenization by splitting on whitespace
    tokens = text.lower().split()
    
    # Truncate if exceeds max length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        
    return tokens




def load_data(name, cache_dir=None):
    """
    Data loading function that downloads .npy files from SocialED_datasets repository.

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : numpy.ndarray
        The loaded dataset.

    Examples
    --------
    >>> from SocialED.utils import load_data
    >>> data = load_data(name='maven') # loads maven.npy dataset
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.socialed/data')
    file_path = os.path.join(cache_dir, name + '.npy')

    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
    else:
        url = "https://github.com/ChenBeici/SocialED_datasets/raw/main/npy_data/" + name + ".npy"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        data = np.load(file_path, allow_pickle=True)
    return data


def logger(epoch=0,
           loss=0,
           score=None,
           target=None,
           time=None,
           verbose=0,
           train=True,
           deep=True):
    """
    Logger for detector.

    Parameters
    ----------
    epoch : int, optional
        The current epoch.
    loss : float, optional
        The current epoch loss value.
    score : torch.Tensor, optional
        The current outlier scores.
    target : torch.Tensor, optional
        The ground truth labels.
    time : float, optional
        The current epoch time.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    train : bool, optional
        Whether the logger is used for training.
    deep : bool, optional
        Whether the logger is used for deep detector.
    """
    if verbose > 0:
        if deep:
            if train:
                print("Epoch {:04d}: ".format(epoch), end='')
            else:
                print("Test: ", end='')

            if isinstance(loss, tuple):
                print("Loss I {:.4f} | Loss O {:.4f} | "
                      .format(loss[0], loss[1]), end='')
            else:
                print("Loss {:.4f} | ".format(loss), end='')

        if verbose > 1:
            if target is not None:
                nmi = eval_nmi(target, score)
                print("NMI {:.4f}".format(nmi), end='')

            if verbose > 2:
                if target is not None:
                    ari = eval_ari(target, score)
                    ami = eval_ami(target, score)

                    print(" | ARI {:.4f} | AMI {:.4f}"
                          .format(ari, ami), end='')

            if time is not None:
                print(" | Time {:.2f}".format(time), end='')

        print()



def pprint(params, offset=0, printer=repr):
    """Pretty print the dictionary 'params'

    Parameters
    ----------
    params : dict
        The dictionary to pretty print
    offset : int, optional
        The offset at the beginning of each line.
    printer : callable, optional
        The function to convert entries to strings, typically
        the builtin str or repr.
    """

    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines



def validate_device(gpu_id):
    """Validate the input GPU ID is valid on the given environment.
    If no GPU is presented, return 'cpu'.

    Parameters
    ----------
    gpu_id : int
        GPU ID to check.

    Returns
    -------
    device : str
        Valid device, e.g., 'cuda:0' or 'cpu'.
    """

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # if gpu is available
    if torch.cuda.is_available():
        # check if gpu id is between 0 and the total number of GPUs
        check_parameter(gpu_id, 0, torch.cuda.device_count(),
                        param_name='gpu id', include_left=True,
                        include_right=False)
        device = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            warnings.warn('The cuda is not available. Set to cpu.')
        device = 'cpu'

    return device


def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, int, float)):
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))

    if not isinstance(low, (numbers.Integral, int, float)):
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))

    if not isinstance(high, (numbers.Integral, int, float)):
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError('Neither low nor high bounds is undefined')

    # if wrong bound values are used
    if low > high:
        raise ValueError(
            'Lower bound > Higher bound')

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (include_left and not include_right) and (
            param < low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and include_right) and (
            param <= low or param > high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))

    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True

