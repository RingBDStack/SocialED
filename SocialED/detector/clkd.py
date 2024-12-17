from time import localtime, strftime, time
import torch.optim as optim
import torch.nn as nn
import json
import argparse
import torch
import dgl
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import metrics
import torch.nn.functional as F
from itertools import combinations
import gensim
import re
import spacy
import pandas as pd
from datetime import datetime
import networkx as nx
from scipy import sparse
from dgl.data.utils import save_graphs, load_graphs
import pickle
from collections import Counter
import en_core_web_lg
import fr_core_news_lg
import sys
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader

from torch.utils.data import Dataset


class args_define:
    def __init__(self, **kwargs):
        # Hyper parameters
        #self.n_epochs = kwargs.get('n_epochs', 5)
        self.n_epochs = kwargs.get('n_epochs', 1)
        self.n_infer_epochs = kwargs.get('n_infer_epochs', 0)
        self.window_size = kwargs.get('window_size', 3)
        self.patience = kwargs.get('patience', 5)
        self.margin = kwargs.get('margin', 3.0)
        self.lr = kwargs.get('lr', 1e-3)
        self.batch_size = kwargs.get('batch_size', 2000)
        self.n_neighbors = kwargs.get('n_neighbors', 800)
        self.word_embedding_dim = kwargs.get('word_embedding_dim', 300)
        self.hidden_dim = kwargs.get('hidden_dim', 8)
        self.out_dim = kwargs.get('out_dim', 32)
        self.num_heads = kwargs.get('num_heads', 4)
        self.use_residual = kwargs.get('use_residual', True)
        self.validation_percent = kwargs.get('validation_percent', 0.1)
        self.test_percent = kwargs.get('test_percent', 0.2)
        self.use_hardest_neg = kwargs.get('use_hardest_neg', False)
        self.metrics = kwargs.get('metrics', 'ami')
        self.use_cuda = kwargs.get('use_cuda', False)
        self.gpuid = kwargs.get('gpuid', 0)
        self.mask_path = kwargs.get('mask_path', None)
        self.log_interval = kwargs.get('log_interval', 10)
        self.is_incremental = kwargs.get('is_incremental', False)
        self.mutual = kwargs.get('mutual', False)
        self.mode = kwargs.get('mode', 0)
        self.add_mapping = kwargs.get('add_mapping', False)
        self.data_path = kwargs.get('data_path', '../model/model_saved/clkd/English')
        self.file_path = kwargs.get('file_path', '../model/model_saved/clkd')
        self.Tmodel_path = kwargs.get('Tmodel_path', '../model/model_saved/clkd/English/Tmodel/')
        self.lang = kwargs.get('lang', 'French')
        self.Tealang = kwargs.get('Tealang', 'English')
        self.t = kwargs.get('t', 1)
        self.data_path1 = kwargs.get('data_path1', '../model/model_saved/clkd/English')
        self.data_path2 = kwargs.get('data_path2', '../model/model_saved/clkd/French')
        self.lang1 = kwargs.get('lang1', 'English')
        self.lang2 = kwargs.get('lang2', 'French')
        self.e = kwargs.get('e', 0)
        self.mt = kwargs.get('mt', 0.5)
        self.rd = kwargs.get('rd', 0.1)
        self.is_static = kwargs.get('is_static', False)
        self.graph_lang = kwargs.get('graph_lang', 'French')
        #self.graph_lang = kwargs.get('graph_lang', 'English')
        self.tgtlang = kwargs.get('tgtlang', 'French')
        self.days = kwargs.get('days', 7)
        #self.initial_lang = kwargs.get('initial_lang', 'French') # DatasetLoader
        self.initial_lang = kwargs.get('initial_lang', 'English')
        self.TransLinear = kwargs.get('TransLinear', True)
        self.TransNonlinear = kwargs.get('TransNonlinear', True)
        # self.tgt = kwargs.get('tgt', 'maven')
        self.tgt = kwargs.get('tgt', 'English')
        self.embpath = kwargs.get('embpath', '../model/model_saved/clkd/dictrans/fr-en-for.npy')
        self.wordpath = kwargs.get('wordpath', '../model/model_saved/clkd/dictrans/wordsFrench.txt')

        # Store all arguments in a single attribute
        self.args = argparse.Namespace(**{
            'n_epochs': self.n_epochs,
            'n_infer_epochs': self.n_infer_epochs,
            'window_size': self.window_size,
            'patience': self.patience,
            'margin': self.margin,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'n_neighbors': self.n_neighbors,
            'word_embedding_dim': self.word_embedding_dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'num_heads': self.num_heads,
            'use_residual': self.use_residual,
            'validation_percent': self.validation_percent,
            'test_percent': self.test_percent,
            'use_hardest_neg': self.use_hardest_neg,
            'metrics': self.metrics,
            'use_cuda': self.use_cuda,
            'gpuid': self.gpuid,
            'mask_path': self.mask_path,
            'log_interval': self.log_interval,
            'is_incremental': self.is_incremental,
            'mutual': self.mutual,
            'mode': self.mode,
            'add_mapping': self.add_mapping,
            'data_path': self.data_path,
            'file_path': self.file_path,
            'Tmodel_path': self.Tmodel_path,
            'lang': self.lang,
            'Tealang': self.Tealang,
            't': self.t,
            'data_path1': self.data_path1,
            'data_path2': self.data_path2,
            'lang1': self.lang1,
            'lang2': self.lang2,
            'e': self.e,
            'mt': self.mt,
            'rd': self.rd,
            'is_static': self.is_static,
            'graph_lang': self.graph_lang,
            'tgtlang': self.tgtlang,
            'days': self.days,
            'initial_lang': self.initial_lang,
            'TransLinear': self.TransLinear,
            'TransNonlinear': self.TransNonlinear,
            'tgt': self.tgt,
            'embpath': self.embpath,
            'wordpath': self.wordpath,
        })




# Inference(prediction)
def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(embedding_save_path, args.data_path,
                                                                                 data_split, train_i, i, args,
                                                                                 args.lang,
                                                                                 args.Tealang)
    # record the time spent in seconds on direct prediction
    time_predict = []
    # Directly predict
    message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    start = time()
    # Infer the representations of all tweets
    extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels, args,
                                                                        labels.device)
    test_nmi = evaluate_model(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i,
                              args.metrics, False)
    seconds_spent = time() - start
    message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    time_predict.append(seconds_spent)
    np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))
    return model


def mutual_infer(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn, metrics,
                 model1, model2, device):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, test_indices1 = getdata(embedding_save_path1,
                                                                                       args.data_path1, data_split1,
                                                                                       train_i, i, args, args.lang1,
                                                                                       args.lang2)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, test_indices2 = getdata(embedding_save_path2,
                                                                                       args.data_path2, data_split2,
                                                                                       train_i, i, args, args.lang2,
                                                                                       args.lang1)

    # model1
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g1, model1, model2, args.lang1,
                                                                               args.lang2,
                                                                               len(labels1), labels1, args, device)
    test_value = evaluate_model(extract_features, extract_labels, test_indices1, -1, num_isolated_nodes2,
                                save_path_i1, args.metrics, False)

    # model2
    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2, args, device)

    test_value = evaluate_model(extract_features, extract_labels, test_indices2, -1, num_isolated_nodes2,
                                save_path_i2, args.metrics, False)
    return model1, model2


def mutual_train(embedding_save_path1, embedding_save_path2, data_split1, data_split2, train_i, i, loss_fn, metrics,
                 device):
    save_path_i1, in_feats1, num_isolated_nodes1, g1, labels1, train_indices1, validation_indices1, test_indices1 = getdata(
        embedding_save_path1, args.data_path1, data_split1, train_i, i, args, args.lang1, args.lang2)
    save_path_i2, in_feats2, num_isolated_nodes2, g2, labels2, train_indices2, validation_indices2, test_indices2 = getdata(
        embedding_save_path2, args.data_path2, data_split2, train_i, i, args, args.lang2, args.lang1)

    model1 = GAT(in_feats1, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
    model2 = GAT(in_feats2, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

    # Optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
    model1_data = {'opt': optimizer1, 'best_value': 1e-9, 'best_epoch': 0,
                   'model': model1, 'peer': model2, 'src': args.lang1, 'tgt': args.lang2,
                   'save_path_i': save_path_i1, 'num_iso_nodes': num_isolated_nodes1, 'g': g1, 'labels': labels1,
                   'train_indices': train_indices1, 'vali_indices': validation_indices1, 'test_indices': test_indices1,
                   'all_vali_nmi': [], 'seconds_train_batches': []}

    model2_data = {'opt': optimizer2, 'best_value': 1e-9, 'best_epoch': 0,
                   'model': model2, 'peer': model1, 'src': args.lang2, 'tgt': args.lang1,
                   'save_path_i': save_path_i2, 'num_iso_nodes': num_isolated_nodes2, 'g': g2, 'labels': labels2,
                   'train_indices': train_indices2, 'vali_indices': validation_indices2, 'test_indices': test_indices2,
                   'all_vali_nmi': [], 'seconds_train_batches': []}
    print("\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n")

    if args.use_cuda:
        model1.to(device)
        model2.to(device)

    for epoch in range(args.n_epochs):

        for model_data in [model1_data, model2_data]:
            losses = []
            total_loss = 0
            for metric in metrics:
                metric.reset()

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                model_data['g'], model_data['train_indices'], sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
            )
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time()
                model_data['model'].train()
                model_data['peer'].eval()

                blocks = [b.to(device) for b in blocks]
                # forward
                pred = model_data['model'](blocks, args)
                batch_nids = blocks[-1].dstdata[dgl.NID].to(device=device, dtype=torch.long)
                batch_labels = model_data['labels'].to(device)[batch_nids]
                peerpred = None

                if args.mode == 2 and epoch >= args.e:
                    if args.add_mapping:
                        peerpred = model_data['peer'](blocks, args, trans=True, src=model_data['src'],
                                                      tgt=model_data['tgt'])
                    else:
                        peerpred = model_data['peer'](blocks, args)
                    peerpred = peerpred.to(device)

                if args.mode == 4 and epoch >= args.e:
                    peerpred = model_data['peer'](blocks, args, trans=True)
                    peerpred = peerpred.to(device)

                loss_outputs = loss_fn(pred, batch_labels, args.rd, peerpred)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                if (args.mode == 2 or args.mode == 4) and epoch >= args.e:
                    l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                    lkd = l(pred, peerpred.to(device))
                    message = "    ".join(["add KD loss", str(loss), str(lkd)])
                    loss = loss + args.mt * lkd
                    print(message)
                    with open(save_path_i1 + '/log.txt', 'a') as f:
                        f.write(message)

                losses.append(loss.item())
                total_loss += loss.item()
                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)
                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices1.shape[0],
                        100. * batch_id / ((train_indices1.shape[0] // args.batch_size) + 1), np.mean(losses))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i1 + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                model_data['opt'].zero_grad()
                loss.backward()
                model_data['opt'].step()
                batch_seconds_spent = time() - start_batch
                model_data['seconds_train_batches'].append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            message += '\n'
            print(message)
            with open(model_data['save_path_i'] + '/log.txt', 'a') as f:
                f.write(message)

            for b in blocks:
                del b
            del pred
            del input_nodes
            del output_nodes
            if peerpred != None:
                del peerpred

            # Validation
            extract_nids, extract_features, extract_labels = extract_embeddings(model_data['g'], model_data['model'],
                                                                                len(model_data['labels']),
                                                                                model_data['labels'],
                                                                                args,
                                                                                device)
            validation_value = evaluate_model(extract_features, extract_labels, model_data['vali_indices'], epoch,
                                              model_data['num_iso_nodes'], model_data['save_path_i'], args.metrics,
                                              True)

            model_data['all_vali_nmi'].append(validation_value)
            if validation_value > model_data['best_value']:
                model_data['best_value'] = validation_value
                model_data['best_epoch'] = epoch
                # Save model
                model_path = model_data['save_path_i'] + '/models'
                if not os.path.isdir(model_path):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model_data['model'].state_dict(), p)
                print(model_data['src'], ':', 'Best model was at epoch ', str(model_data['best_epoch']))

            for metric in metrics:
                metric.reset()

    with open(save_path_i1 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch ' + str(model1_data['best_epoch'])
        f.write(message)
    with open(save_path_i2 + '/evaluate.txt', 'a') as f:
        message = 'Best model was at epoch ' + str(model2_data['best_epoch'])
        f.write(message)
    # Save all validation nmi
    np.save(save_path_i1 + '/all_vali_nmi.npy', np.asarray(model1_data['all_vali_nmi']))
    np.save(save_path_i2 + '/all_vali_nmi.npy', np.asarray(model2_data['all_vali_nmi']))
    # save all seconds_train
    np.save(save_path_i1 + '/seconds_train_batches.npy', np.asarray(model1_data['seconds_train_batches']))
    np.save(save_path_i2 + '/seconds_train_batches.npy', np.asarray(model2_data['seconds_train_batches']))

    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g1, model1, model2, args.lang1,
                                                                               args.lang2,
                                                                               len(labels1), labels1, args, device)
    test_value = evaluate_model(extract_features, extract_labels, test_indices1, -1, num_isolated_nodes1,
                                save_path_i1, args.metrics, False)

    extract_nids, extract_features, extract_labels = mutual_extract_embeddings(g2, model2, model1, args.lang2,
                                                                               args.lang1,
                                                                               len(labels2), labels2, args, device)
    test_value = evaluate_model(extract_features, extract_labels, test_indices2, -1, num_isolated_nodes2,
                                save_path_i2, args.metrics, False)

    return model1, model2


# Train on initial/maintenance graphs
def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None):
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, args.data_path, data_split, train_i, i, args, args.lang, args.Tealang)

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        if args.use_cuda:
            model.cuda()

    if args.mode == 2 or args.mode == 4:
        Tmodel = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        Tmodel_path = args.Tmodel_path + '/block_' + str(train_i) + '/models/best.pt'
        Tmodel.load_state_dict(torch.load(Tmodel_path))
        if args.use_cuda:
            Tmodel.cuda()
        Tmodel.eval()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Start training
    message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_nmi = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_nmi = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        Tpred = None
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            start_batch = time()
            model.train()
            # forward
            blocks = [b.to(train_indices.device) for b in blocks]
            pred = model(blocks, args)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
            if args.mode == 2:
                if args.add_mapping:
                    Tpred = Tmodel(blocks, args, trans=True, src=args.lang, tgt=args.Tealang)
                else:
                    Tpred = Tmodel(blocks, args)
            if args.mode == 4:
                Tpred = Tmodel(blocks, args, trans=True)

            batch_nids = blocks[-1].dstdata[dgl.NID].to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss_outputs = loss_fn(pred, batch_labels, args.rd, Tpred)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            if args.mode == 2 or args.mode == 4:
                # p = torch.matmul(pred,pred.T)
                # Tp = torch.matmul(Tpred,Tpred.T)
                # kl = F.kl_div(p.softmax(dim=-1).log(), Tp.softmax(dim=-1), reduction='sum')
                l = nn.L1Loss(size_average=True, reduce=True, reduction='average')
                # l = torch.nn.MSELoss(reduce=True, size_average=True)
                lkd = l(pred, Tpred)
                message = "    ".join(["add KD loss", str(loss), str(lkd)])
                print(message)
                loss = loss + args.mt * lkd
            losses.append(loss.item())
            total_loss += loss.item()

            for metric in metrics:
                metric(pred, batch_labels, loss_outputs)

            if batch_id % args.log_interval == 0:
                message += 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_id * args.batch_size, train_indices.shape[0],
                    100. * batch_id / ((train_indices.shape[0] // args.batch_size) + 1), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                losses = []

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_seconds_spent = time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch
            del pred
            if args.mode != 0:
                del Tpred
            for b in blocks:
                del b

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels, args,
                                                                            labels.device)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        validation_nmi = evaluate_model(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                        save_path_i, args.metrics, True)
        all_vali_nmi.append(validation_nmi)

        # Early stop
        if validation_nmi > best_vali_nmi:
            best_vali_nmi = validation_nmi
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print('Best model saved after epoch ', str(epoch))
        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
    # Save time spent on epochs
    np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
    print('Saved mins_train_epochs.')
    # Save time spent on batches
    np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
    print('Saved seconds_train_batches.')
    # Load the best model of the current block

    best_model_path = save_path_i + '/models/best.pt'
    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded.")
    return model


# utiles
# 将整个图数据集划分为训练集、验证集和测试集，并将这些集合的索引存储在给定的路径（如果提供了路径）中。
def generateMasks(length, data_split, train_i, i, validation_percent=0.1, test_percent=0.2, save_path=None):
    # verify total number of nodes
    assert length == data_split[i]
    if train_i == i:
        # randomly shuffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices, save_path + '/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return train_indices, validation_indices, test_indices
    # If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices


def getdata(embedding_save_path, data_path, data_split, train_i, i, args, src=None, tgt=None):
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)
    # load data
    data = SocialDataset(data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]  # feature dimension

    g = dgl.DGLGraph(data.matrix,
                     readonly=True)
    num_isolated_nodes = graph_statistics(g, save_path_i)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly(readonly_state=True)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    g = g.to(device)

    mask_path = save_path_i + '/masks'
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    if train_i == i:
        train_indices, validation_indices, test_indices = generateMasks(len(labels), data_split, train_i, i,
                                                                        args.validation_percent,
                                                                        args.test_percent,
                                                                        mask_path)
    else:
        test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent,
                                     args.test_percent,
                                     mask_path)
    if args.use_cuda:
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        if train_i == i:
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()
    # features = F.normalize(features, p=2, dim=1)

    g.ndata['h'] = features
    if args.mode == 4:
        tranfeatures = np.load(
            data_path + '/' + str(i) + '/' + "-".join([src, tgt, 'features']) + '.npy')
        tranfeatures = torch.FloatTensor(tranfeatures)
        # tranfeatures = F.normalize(tranfeatures, p=2, dim=1)
        if args.use_cuda:
            tranfeatures = tranfeatures.cuda()
        g.ndata['tranfeatures'] = tranfeatures

    if train_i == i:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices
    else:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices


# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels, args, device):
    with torch.no_grad():
        model.eval()
        select_indices = torch.LongTensor(range(0, num_all_samples)).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, select_indices, sampler,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
        )
        labels = labels.cpu().detach().float()
        fea_list = []
        nid_list = []
        label_list = []
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            extract_features = model(blocks, args).float()
            extract_features = extract_features.cpu().detach()
            extract_nids = blocks[-1].dstdata[dgl.NID].data.cpu()  # node ids
            extract_labels = labels[extract_nids].float()  # labels of all nodes
            fea_list.append(extract_features.numpy())
            nid_list.append(extract_nids.numpy())
            label_list.append(extract_labels.numpy())

        extract_features = np.concatenate(fea_list, axis=0).astype(np.float32)
        extract_labels = np.concatenate(label_list, axis=0).astype(np.float32)
        extract_nids = np.concatenate(nid_list, axis=0)

    return (extract_nids, extract_features, extract_labels)


def mutual_extract_embeddings(g, model, peer, src, tgt, num_all_samples, labels, args, device):
    with torch.no_grad():
        model.eval()
        peer.eval()
        select_indices = torch.LongTensor(range(0, num_all_samples)).to(device)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g, select_indices, sampler,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
        )
        fea_list = []
        nid_list = []
        label_list = []
        labels = labels.cpu().detach()
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            extract_features1 = model(blocks, args)
            if (args.mode == 2 and args.add_mapping):
                print("** add linear tran peer feature **", src, tgt)
                extract_features2 = peer(blocks, args, True, src=src, tgt=tgt)  # representations of all nodes
            elif args.mode == 4:
                print("** add nonlinear tran peer feature **", src, tgt)
                extract_features2 = peer(blocks, args, True)  # representations of all nodes
            else:
                print("** add feature **")
                extract_features2 = peer(blocks, args)
            extract_nids = blocks[-1].dstdata[dgl.NID].cpu()
            extract_labels = labels[extract_nids]  # labels of all nodes
            extract_features1 = extract_features1.cpu().detach()
            extract_features2 = extract_features2.cpu().detach()
            extract_features = torch.cat((extract_features1, extract_features2), 1).numpy()
            # extract_features = extract_features1.numpy()
            fea_list.append(extract_features)
            nid_list.append(extract_nids.numpy())
            label_list.append(extract_labels.numpy())

        for b in blocks:
            del b
        del input_nodes, output_nodes, select_indices

        # assert batch_id == 0
        # extract_nids = extract_nids.data.cpu().numpy()
        # extract_features1 = extract_features1.data.cpu().detach()
        # extract_features2 = extract_features2.data.cpu().detach()
        # extract_features = torch.cat((extract_features1,extract_features2),1).numpy()
        # extract_labels = extract_labels.data.cpu().detach().numpy()
        extract_features = np.concatenate(fea_list, axis=0)
        extract_labels = np.concatenate(label_list, axis=0)
        extract_nids = np.concatenate(nid_list, axis=0)
        # generate train/test mask
        A = np.arange(num_all_samples)
        # print("A", A)
        # assert (A == extract_nids).all()

    return (extract_nids, extract_features, extract_labels)


def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, metric, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes
    n_classes = len(set(list(labels_true)))

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:", nmi, 'ami:', ami, 'ari:', ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if metric == 'ari':
        print('use ari')
        value = ari
    if metric == 'ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)


def evaluate_model(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, metrics,
                   is_validation=True,
                   file_name='evaluate.txt'):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, metrics)
    if is_validation:
        split = 'validation'
    else:
        split = 'test'
    message += '\tNumber of ' + split + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + split + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + split + ' '
    message += metrics + ': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, metrics,
                                                save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + split + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + split + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + split + f' {metrics}: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    print("*********************************")
    with open(save_path + f'/{file_name}', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI " + str(NMI) + " AMI " + str(AMI) + ' ARI ' + str(ARI))
    print(message)

    return value


# metrics


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target, rd, peer_embeddings=None):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        if peer_embeddings != None:
            peer_ap_distances = (peer_embeddings[triplets[:, 0]] - peer_embeddings[triplets[:, 1]]).pow(2).sum(1)
            peer_an_distances = (peer_embeddings[triplets[:, 0]] - peer_embeddings[triplets[:, 2]]).pow(2).sum(1)
            kd_ap_losses = F.relu(-peer_ap_distances + ap_distances)
            kd_an_losses = F.relu(-an_distances + peer_an_distances)
            print("losses.mean():", losses.mean(), "ap_mean:", kd_ap_losses.mean(), "an_mean:", kd_an_losses.mean())
            return losses.mean() + rd * kd_ap_losses.mean() + rd * kd_an_losses.mean(), len(triplets)
        else:
            return losses.mean(), len(triplets)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


'''
def hard_negative_and_positive(loss_values1, loss_values2):
    hard_negatives = np.where(loss_values1 > 0)[0]

    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None
'''


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                              negative_selection_fn=hardest_negative,
                                                                                              cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


# model
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['h'].float()  # 确保 h 为 Float 类型
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]
        blocks[layer_id].dstdata['z'] = z_dst

        blocks[layer_id].apply_edges(self.edge_attention)
        blocks[layer_id].update_all(
            self.message_func,
            self.reduce_func)

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id).float() for attn_head in self.heads]  # 确保 head_outs 为 Float 类型
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, blocks, args, trans=False, src=None, tgt=None):
        print("Entering forward function")
        if trans:
            if args.mode == 4:
                features = blocks[0].srcdata['tranfeatures'].float()
                print("This is nonlinear trans!")
                blocks[0].srcdata['h'] = features
            if args.mode == 2 and args.add_mapping:
                features = blocks[0].srcdata['h'].cpu().detach().float()
                W = torch.from_numpy(
                    torch.load(
                        args.file_path + '/LinearTranWeight/spacy_{}_{}/best_mapping.pth'.format(src, tgt))).float()
                print("This is linear trans!")
                part1 = torch.index_select(features, 1, torch.tensor(range(0, args.word_embedding_dim)))
                part1 = torch.matmul(part1, torch.FloatTensor(W))
                part2 = torch.index_select(features, 1,
                                           torch.tensor(range(args.word_embedding_dim, features.size()[1])))
                features = torch.cat((part1, part2), 1).cuda()
                blocks[0].srcdata['h'] = features

        h = self.layer1(blocks, 0).float()
        h = F.elu(h)
        blocks[1].srcdata['h'] = h
        h = self.layer2(blocks, 1).float()
        return h


class Arabic_preprocessor:
    def __init__(self, tokenizer, **cfg):
        self.tokenizer = tokenizer

    def clean_text(self, text):
        search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
                  '&quot;', '?', '؟', '!']
        replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ',
                   ' ؟ ', ' ! ']

        # remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(p_tashkeel, "", text)

        # remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        text = re.sub(p_longation, subst, text)

        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('اا', 'ا')

        for i in range(len(search)):
            text = text.replace(search[i], replace[i])

        # trim
        text = text.strip()

        return text

    def __call__(self, text):
        preprocessed = self.clean_text(text)
        return self.tokenizer(preprocessed)


class CLKD:
    def __init__(self, args):
        self.embedding_save_path1 = None
        self.embedding_save_path2 = None
        self.embedding_save_path = None
        self.data_split1 = None
        self.data_split2 = None
        self.data_split = None
        self.args = args

    def preprocess(self):
        preprocessor = Preprocessor(self.args)
        preprocessor.generate_initial_features()
        preprocessor.construct_graph()

        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.args.gpuid)
            self.device = torch.device("cuda:{}".format(self.args.gpuid))
        else:
            self.device = torch.device('cpu')

        if self.args.mutual:
            print("args.mutual is true")
            path1 = os.path.join(self.args.data_path1, f"{self.args.mode}mode")
            path2 = os.path.join(self.args.data_path2, f"{self.args.mode}mode")
            os.makedirs(path1, exist_ok=True)
            os.makedirs(path2, exist_ok=True)

            timestamp = strftime("%m%d%H%M%S", localtime())
            self.embedding_save_path1 = os.path.join(path1,
                                                     f'embeddings_{timestamp}-{self.args.mode}-{self.args.lang2}')
            self.embedding_save_path2 = os.path.join(path2,
                                                     f'embeddings_{timestamp}-{self.args.mode}-{self.args.lang1}')

            if not self.args.add_mapping and (self.args.mode in [0, 1, 2]):
                self.embedding_save_path1 += "-nomap"
                self.embedding_save_path2 += "-nomap"
            else:
                self.embedding_save_path1 += "-map"
                self.embedding_save_path2 += "-map"

            os.makedirs(self.embedding_save_path1, exist_ok=True)
            os.makedirs(self.embedding_save_path2, exist_ok=True)

            print("embedding_save_path1 and embedding_save_path2: ", self.embedding_save_path1,
                  self.embedding_save_path2)
            with open(os.path.join(self.embedding_save_path1, 'args.txt'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=2)
            with open(os.path.join(self.embedding_save_path2, 'args.txt'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=2)

            self.data_split1 = np.load(os.path.join(self.args.data_path1, 'data_split.npy'))
            self.data_split2 = np.load(os.path.join(self.args.data_path2, 'data_split.npy'))
            print("data_split1:", self.data_split1, 'data_split2:', self.data_split2)
        else:
            embedding_dir = os.path.join(self.args.data_path, f'{self.args.mode}mode')
            os.makedirs(embedding_dir, exist_ok=True)

            timestamp = strftime("%m%d%H%M%S", localtime())
            self.embedding_save_path = os.path.join(embedding_dir,
                                                    f'embeddings_{timestamp}-{self.args.mode}-{self.args.Tealang}')

            if not self.args.add_mapping and (self.args.mode in [0, 1, 2]):
                self.embedding_save_path += "-nomap"
            else:
                self.embedding_save_path += "-map"

            os.makedirs(self.embedding_save_path, exist_ok=True)

            print("embedding_save_path: ", self.embedding_save_path)
            with open(os.path.join(self.embedding_save_path, 'args.txt'), 'w') as f:
                json.dump(self.args.__dict__, f, indent=2)

            self.data_split = np.load(os.path.join(self.args.data_path, 'data_split.npy'))

    def fit(self):
        # 初始化损失函数和度量指标
        if self.args.use_hardest_neg:
            loss_fn = OnlineTripletLoss(self.args.margin, HardestNegativeTripletSelector(self.args.margin))
        else:
            loss_fn = OnlineTripletLoss(self.args.margin, RandomNegativeTripletSelector(self.args.margin))

        metrics = [AverageNonzeroTripletsMetric()]
        self.train_i = 0

        if self.args.mutual:
            self.model1, self.model2 = mutual_train(self.embedding_save_path1, self.embedding_save_path2,
                                                    self.data_split1, self.data_split2, self.train_i, 0,
                                                    loss_fn, metrics, self.device)
        else:
            self.model = initial_maintain(self.train_i, 0, self.data_split, metrics, self.embedding_save_path, loss_fn,
                                          None)

    def detection(self):
        if self.args.use_hardest_neg:
            loss_fn = OnlineTripletLoss(self.args.margin, HardestNegativeTripletSelector(self.args.margin))
        else:
            loss_fn = OnlineTripletLoss(self.args.margin, RandomNegativeTripletSelector(self.args.margin))

        metrics = [AverageNonzeroTripletsMetric()]

        if self.args.mutual:
            self.model1, self.model2 = mutual_infer(self.embedding_save_path1, self.embedding_save_path2,
                                                    self.data_split1, self.data_split2,
                                                    self.train_i, 0, loss_fn, metrics, self.model1, self.model2,
                                                    self.device)
            if self.args.is_incremental:
                for i in range(1, min(self.data_split1.shape[0], self.data_split2.shape[0])):
                    print("enter i ", str(i))
                    self.model1, self.model2 = mutual_infer(self.embedding_save_path1, self.embedding_save_path2,
                                                            self.data_split1, self.data_split2,
                                                            self.train_i, i, loss_fn, metrics, self.model1, self.model2,
                                                            self.device)
                    if i % self.args.window_size == 0:
                        self.train_i = i
                        self.model1, self.model2 = mutual_train(self.embedding_save_path1, self.embedding_save_path2,
                                                                self.data_split1, self.data_split2, self.train_i, i,
                                                                loss_fn, metrics, self.device)

            # 最终预测并保存预测结果
            data1 = SocialDataset(self.args.data_path1, 0)
            g1 = dgl.DGLGraph(data1.matrix)
            labels1 = torch.LongTensor(data1.labels)
            data2 = SocialDataset(self.args.data_path2, 0)
            g2 = dgl.DGLGraph(data2.matrix)
            labels2 = torch.LongTensor(data2.labels)

            self.mutual_detection_path1 = self.args.file_path + '/mutual_detection_split1/'
            os.makedirs(self.mutual_detection_path1, exist_ok=True)

            train_indices1, validation_indices1, test_indices1 = generateMasks(len(labels), self.data_split1,
                                                                               self.train_i, 0,
                                                                               0.1, 0.2, self.mutual_detection_path)
            g1.ndata['h'] = torch.tensor(data1.features)

            self.mutual_detection_path2 = self.args.file_path + '/mutual_detection_split2/'
            os.makedirs(self.mutual_detection_path2, exist_ok=True)
            train_indices2, validation_indices2, test_indices2 = generateMasks(len(labels), self.data_split2,
                                                                               self.train_i, 0,
                                                                               0.1, 0.2, self.mutual_detection_path)
            g2.ndata['h'] = torch.tensor(data2.features)

            _, extract_features1, _ = mutual_extract_embeddings(g1, self.model1, self.model2, self.args.lang1,
                                                                self.args.lang2, len(labels1), labels1, self.args,
                                                                self.device)
            _, extract_features2, _ = mutual_extract_embeddings(g2, self.model2, self.model1, self.args.lang2,
                                                                self.args.lang1, len(labels2), labels2, self.args,
                                                                self.device)

            predictions1 = []
            ground_truths1 = []
            # Extract labels
            test_indices1 = torch.load(self.mutual_detection_path1 + '/test_indices.pt')
            labels_true1 = extract_labels[test_indices1]
            # Extract features
            X1 = extract_features[test_indices1, :]
            assert labels_true1.shape[0] == X1.shape[0]
            # Get the total number of classes
            n_classes1 = len(set(list(labels_true1)))
            # kmeans clustering
            kmeans1 = KMeans(n_clusters=n_classes1, random_state=0).fit(X1)
            predictions1 = kmeans1.labels_
            ground_truths1 = labels_true1

            predictions2 = []
            ground_truths2 = []
            # Extract labels
            test_indices2 = torch.load(self.mutual_detection_path2 + '/test_indices.pt')
            labels_true2 = extract_labels[test_indices2]
            # Extract features
            X2 = extract_features[test_indices2, :]
            assert labels_true2.shape[0] == X2.shape[0]
            # Get the total number of classes
            n_classes2 = len(set(list(labels_true2)))
            # kmeans clustering
            kmeans2 = KMeans(n_clusters=n_classes2, random_state=0).fit(X2)
            predictions2 = kmeans2.labels_
            ground_truths2 = labels_true2

            return predictions1, ground_truths1, predictions2, ground_truths2

        else:
            self.model = initial_maintain(self.train_i, 0, self.data_split, metrics, self.embedding_save_path, loss_fn,
                                          self.model)
            if self.args.is_incremental:
                for i in range(1, self.data_split.shape[0]):
                    print("incremental setting")
                    print("enter i ", str(i))
                    self.model = infer(self.train_i, i, self.data_split, metrics, self.embedding_save_path, loss_fn,
                                       self.model)
                    if i % self.args.window_size == 0:
                        self.model = initial_maintain(self.train_i, i, self.data_split, metrics,
                                                      self.embedding_save_path, loss_fn, self.model)

            # 最终预测并保存预测结果
            data = SocialDataset(self.args.data_path, 0)
            g = dgl.DGLGraph(data.matrix)
            labels = torch.LongTensor(data.labels)

            predictions = []
            ground_truths = []
            self.detection_path = self.args.file_path + '/detection_split/'
            os.makedirs(self.detection_path, exist_ok=True)

            train_indices, validation_indices, test_indices = generateMasks(len(labels), self.data_split, self.train_i,
                                                                            0,
                                                                            0.1, 0.2, self.detection_path)

            g.ndata['h'] = torch.tensor(data.features)  # Assuming data.features contains the feature data

            _, extract_features, extract_labels = extract_embeddings(g, self.model, len(labels), labels, self.args,
                                                                     self.device)

            # Extract labels
            test_indices = torch.load(self.detection_path + '/test_indices.pt')

            labels_true = extract_labels[test_indices]
            # Extract features
            X = extract_features[test_indices, :]
            assert labels_true.shape[0] == X.shape[0]
            n_test_tweets = X.shape[0]

            # Get the total number of classes
            n_classes = len(set(list(labels_true)))

            # kmeans clustering
            kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
            predictions = kmeans.labels_
            ground_truths = labels_true

            return predictions, ground_truths

    def evaluate(self, predictions, ground_truths, predictions2=None, ground_truths2=None):

        if self.args.mutual:

            ars1, ami1, nmi1 = self.evaluate2(predictions1, ground_truths1)
            print(f"Model1 Adjusted Rand Index (ARI): {ars1}")
            print(f"Model1 Adjusted Rand Index (AMI): {ami1}")
            print(f"Model1 Adjusted Rand Index (NMI): {nmi1}")
            ars2, ami2, nmi2 = self.evaluate2(predictions2, ground_truths2)
            print(f"Model2 Adjusted Rand Index (ARI): {ars2}")
            print(f"Model2 Adjusted Rand Index (AMI): {ami2}")
            print(f"Model2 Adjusted Rand Index (NMI): {nmi2}")

        else:
            ars, ami, nmi = self.evaluate2(predictions, ground_truths)
            print(f"Model Adjusted Rand Index (ARI): {ars}")
            print(f"Model Adjusted Mutual Information (AMI): {ami}")
            print(f"Model Normalized Mutual Information (NMI): {nmi}")

        # logging.info(f"Adjusted Rand Index (ARI): {ars}")
        # logging.info(f"Adjusted Mutual Information (AMI): {ami}")
        # logging.info(f"Normalized Mutual Information (NMI): {nmi}")
        return ars, ami, nmi

    def evaluate2(self, predictions, ground_truths):
        ars = metrics.adjusted_rand_score(ground_truths, predictions)

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        return ars, ami, nmi

class Preprocessor:
    def __init__(self, args):
        self.device = None
        self.args = args

    def generate_initial_features(self):
        # self.args = self.self.args
        save_path = self.args.file_path + '/features/'
        os.makedirs(save_path, exist_ok=True)
        #print(self.args.initial_lang) wasd
        if self.args.initial_lang == "French":
            df = DatasetLoader('event2018').load_data()
        elif self.args.initial_lang == "Arabic":
            df = DatasetLoader('arabic_twitter').load_data()
        elif self.args.initial_lang == "English":
            df = DatasetLoader('event2012').load_data()
        #elif self.args.initial_lang == "Spanish":
            #df = DatasetLoader('maven').load_data()
        else:
            # print(self.args.initial_lang)
            raise NotImplementedError("Language not supported")

        df = df[['event_id', 'words', 'filtered_words', 'created_at']].copy()
        print("Loaded {} data, shape {}".format(self.args.initial_lang, df.shape))
        print(df.head(10))

        t_features = self.df_to_t_features(df)
        print("Time features generated.")
        d_features = self.documents_to_features(df, self.args.initial_lang)
        print("Original document features generated")

        combined_features = np.concatenate((d_features, t_features), axis=1)
        print("Concatenated document features and time features.")
        np.save(os.path.join(save_path, 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(
            self.args.initial_lang)),
                combined_features)

        if self.args.TransLinear:
            dl_features = self.getlinear_transform_features(d_features, self.args.initial_lang, self.args.tgt)
            lcombined_features = np.concatenate((dl_features, t_features), axis=1)
            print("Linear transformed features generated")
            np.save(os.path.join(save_path, 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}_{}.npy'.format(
                self.args.initial_lang, self.args.tgt)),
                    lcombined_features)

        ''' 
        if self.args.TransNonlinear:
            dnl_features = nonlinear_transform_features(self.args.wordpath,self.args.embpath,df)   
            dlcombined_features = np.concatenate((dnl_features, t_features), axis=1)
            print(dlcombined_features)
            print("Nonlinear trans features generated.")
            np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_nonlinear-{}_{}.npy'.format(self.args.initial_lang, self.args.tgt),
                dlcombined_features)
        '''

    def documents_to_features(self, df, initial_lang):
        if initial_lang == "French":
            nlp = spacy.load("fr_core_news_lg")
        elif initial_lang == "Arabic":
            nlp = spacy.load('spacy.arabic.model')
            nlp.tokenizer = Arabic_preprocessor(nlp.tokenizer)
        elif initial_lang == "English":
            nlp = spacy.load("en_core_web_lg")
        # elif initial_lang == "Spanish":
            # nlp = spacy.load("fr_core_news_lg")
        else:
            raise ValueError("Language not supported")

        features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector if len(x) != 0 else nlp(' ').vector).values
        return np.stack(features, axis=0)

    def get_word2id_emb(self, wordpath, embpath):
        word2id = {}
        with open(wordpath, 'r') as f:
            for i, w in enumerate(list(f.readlines()[0].split())):
                word2id[w] = i
        embeddings = np.load(embpath)
        return word2id, embeddings

    def nonlinear_transform_features(self, wordpath, embpath, df):
        word2id, embeddings = self.get_word2id_emb(wordpath, embpath)
        features = df.filtered_words.apply(lambda x: [embeddings[word2id[w]] for w in x if w in word2id])
        f_list = []
        for f in features:
            if len(f) != 0:
                f_list.append(np.mean(f, axis=0))
            else:
                f_list.append(np.zeros((300,)))
        return np.stack(f_list, axis=0)

    def getlinear_transform_features(self, features, src, tgt):
        W = torch.load(self.args.file_path + "/LinearTranWeight/spacy_{}_{}/best_mapping.pth".format(src, tgt))
        return np.matmul(features, W)

    def extract_time_feature(self, t_str):
        t = datetime.fromisoformat(str(t_str))
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = t - OLE_TIME_ZERO
        return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]

    def df_to_t_features(self, df):
        return np.asarray([self.extract_time_feature(t_str) for t_str in df['created_at']])

    def construct_graph(self):
        # create save path
        if self.args.is_static:
            save_path = self.args.file_path + "/hash_static-{}-{}/".format(str(self.args.days), self.args.graph_lang)
        else:
            save_path = self.args.file_path + "/{}/".format(self.args.graph_lang)

        os.makedirs(save_path, exist_ok=True)

        # load df data
        if self.args.graph_lang == "French":
            df = DatasetLoader("maven").load_data()
        elif self.args.graph_lang == "Arabic":
            df = DatasetLoader("arabic_twitter").load_data()
            name2id = {}
            for id, name in enumerate(df['event_id'].unique()):
                name2id[name] = id
            print(name2id)
            df['event_id'] = df['event_id'].apply(lambda x: name2id[x])
            df.drop_duplicates(['tweet_id'], inplace=True, keep='first')

        elif self.args.graph_lang == "English":
            df = DatasetLoader("event2012").load_data()
        #elif self.args.graph_lang == "Spanish":
            #df = DatasetLoader('maven').load_data()
        print("{} Data converted to dataframe.".format(self.args.graph_lang))

        # sort data by time
        df = df.sort_values(by='created_at').reset_index()
        # append date
        df['date'] = [d.date() for d in df['created_at']]

        nf = None
        # load features
        f = np.load(self.args.file_path + '/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(
            self.args.graph_lang))
        nonleafilename = self.args.file_path + "/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}_{}.npy".format(
            self.args.graph_lang, self.args.tgtlang)
        nf = np.load(nonleafilename)

        # construct graph
        message, data_split, all_graph_mins = self.construct_incremental_dataset(self.args, df, save_path, f, nf, False)
        with open(save_path + "node_edge_statistics.txt", "w") as text_file:
            text_file.write(message)
        np.save(save_path + 'data_split.npy', np.asarray(data_split))
        print("Data split: ", data_split)
        np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
        print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

    def construct_graph_from_df(self, df, G=None):
        if G is None:
            G = nx.Graph()
        for _, row in df.iterrows():
            tid = 't_' + str(row['tweet_id'])
            G.add_node(tid)
            G.nodes[tid]['tweet_id'] = True

            user_ids = row['user_mentions']
            user_ids.append(row['user_id'])
            user_ids = ['u_' + str(each) for each in user_ids]
            G.add_nodes_from(user_ids)
            for each in user_ids:
                G.nodes[each]['user_id'] = True

            entities = row['entities']
            entities = ['e_' + str(each) for each in entities]
            G.add_nodes_from(entities)
            for each in entities:
                G.nodes[each]['entity'] = True

            hashtags = row['hashtags']
            hashtags = ['h_' + str(each) for each in hashtags]
            G.add_nodes_from(hashtags)
            for each in hashtags:
                G.nodes[each]['hashtag'] = True

            edges = []
            edges += [(tid, each) for each in user_ids]
            edges += [(tid, each) for each in entities]
            edges += [(tid, each) for each in hashtags]
            G.add_edges_from(edges)

        return G

    # convert networkx graph to dgl graph and store its sparse binary adjacency matrix
    def networkx_to_dgl_graph(self, G, save_path=None):
        message = ''
        print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
        message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
        all_start = time()

        print('\tGetting a list of all nodes ...')
        message += '\tGetting a list of all nodes ...\n'
        start = time()
        all_nodes = list(G.nodes)
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tGetting adjacency matrix ...')
        message += '\tGetting adjacency matrix ...\n'
        start = time()
        A = nx.to_numpy_array(G)  # Returns the graph adjacency matrix as a NumPy matrix.
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # compute commuting matrices
        print('\tGetting lists of nodes of various types ...')
        message += '\tGetting lists of nodes of various types ...\n'
        start = time()
        tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
        userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
        hash_nodes = list(nx.get_node_attributes(G, 'hashtag').keys())
        entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
        del G
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tConverting node lists to index lists ...')
        message += '\tConverting node lists to index lists ...\n'
        start = time()
        indices_tid = [all_nodes.index(x) for x in tid_nodes]
        indices_userid = [all_nodes.index(x) for x in userid_nodes]
        indices_hashtag = [all_nodes.index(x) for x in hash_nodes]
        indices_entity = [all_nodes.index(x) for x in entity_nodes]
        del tid_nodes
        del userid_nodes
        del hash_nodes
        del entity_nodes
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # tweet-user-tweet
        print('\tStart constructing tweet-user-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-user matrix ...')
        message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
        start = time()
        w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_userid = sparse.csr_matrix(w_tid_userid)
        del w_tid_userid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_userid_tid = s_w_tid_userid.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-user * user-tweet ...')
        message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
        start = time()
        s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
            print("Sparse binary userid commuting matrix saved.")
            del s_m_tid_userid_tid
        del s_w_tid_userid
        del s_w_userid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # tweet-ent-tweet
        print('\tStart constructing tweet-ent-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-ent matrix ...')
        message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
        start = time()
        w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
        del w_tid_entity
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_entity_tid = s_w_tid_entity.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-ent * ent-tweet ...')
        message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
        start = time()
        s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
            print("Sparse binary entity commuting matrix saved.")
            del s_m_tid_entity_tid
        del s_w_tid_entity
        del s_w_entity_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # tweet-hashtag-tweet
        print('\tStart constructing tweet-hashtag-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-hashtag matrix ...')
        message += '\tStart constructing tweet-hashtag-tweet commuting matrix ...\n\t\t\tStart constructing tweet-hashtag matrix ...\n'
        start = time()
        w_tid_hash = A[np.ix_(indices_tid, indices_hashtag)]
        del A
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_hash = sparse.csr_matrix(w_tid_hash)
        del w_tid_hash
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_hash_tid = s_w_tid_hash.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-hashtag * hashtag-tweet ...')
        message += '\t\t\tCalculating tweet-hashtag * hashtag-tweet ...\n'
        start = time()
        s_m_tid_hash_tid = s_w_tid_hash * s_w_hash_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_hash_tid.npz", s_m_tid_hash_tid)
            print("Sparse binary hashtag commuting matrix saved.")
            del s_m_tid_hash_tid
        del s_w_tid_hash
        del s_w_hash_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # compute tweet-tweet adjacency matrix
        print('\tComputing tweet-tweet adjacency matrix ...')
        message += '\tComputing tweet-tweet adjacency matrix ...\n'
        start = time()
        if save_path is not None:
            s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
            print("Sparse binary userid commuting matrix loaded.")
            s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
            print("Sparse binary entity commuting matrix loaded.")
            s_m_tid_hash_tid = sparse.load_npz(save_path + "s_m_tid_hash_tid.npz")
            print("Sparse binary hashtag commuting matrix loaded.")

        s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
        del s_m_tid_userid_tid
        del s_m_tid_entity_tid
        s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_hash_tid).astype('bool')
        del s_m_tid_hash_tid
        del s_A_tid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        all_mins = (time() - all_start) / 60
        print('\tOver all time elapsed: ', all_mins, ' mins\n')
        message += '\tOver all time elapsed: '
        message += str(all_mins)
        message += ' mins\n'

        if save_path is not None:
            sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
            print("Sparse binary adjacency matrix saved.")
            s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
            print("Sparse binary adjacency matrix loaded.")

        # create corresponding dgl graph
        G = dgl.DGLGraph(s_bool_A_tid_tid)
        print('We have %d nodes.' % G.number_of_nodes())
        print('We have %d edges.' % G.number_of_edges())
        message += 'We have '
        message += str(G.number_of_nodes())
        message += ' nodes.'
        message += 'We have '
        message += str(G.number_of_edges())
        message += ' edges.\n'

        return all_mins, message

    def construct_incremental_dataset(self, args, df, save_path, features, nfeatures, test=False):
        data_split = []
        all_graph_mins = []
        message = ""
        # extract distinct dates
        distinct_dates = df.date.unique()
        # print("Distinct dates: ", distinct_dates)
        print("Number of distinct dates: ", len(distinct_dates))
        message += "Number of distinct dates: "
        message += str(len(distinct_dates))
        message += "\n"
        print("Start constructing initial graph ...")
        message += "\nStart constructing initial graph ...\n"

        if self.args.is_static:
            ini_df = df.loc[df['date'].isin(distinct_dates[:self.args.days])]
            days = self.args.days
        else:
            ini_df = df.loc[df['date'].isin(distinct_dates[:1])]
            days = 1

        print("Initial graph contains %d days" % days)
        message += "Initial graph contains %d days\n" % days

        path = save_path + '0/'
        if not os.path.exists(path):
            os.mkdir(path)

        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))

        G = self.construct_graph_from_df(ini_df)
        grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
        message += graph_message
        print("Initial graph saved")
        message += "Initial graph saved\n"
        # record the total number of tweets
        data_split.append(ini_df.shape[0])
        # record the time spent for graph conversion
        all_graph_mins.append(grap_mins)
        # extract and save the labels of corresponding tweets
        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))
        np.save(path + 'df.npy', ini_df)
        # ini_df['created_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        np.save(path + "time.npy", ini_df['created_at'].values)
        print("Labels and times saved.")
        message += "Labels and times saved.\n"
        # extract and save the features of corresponding tweets
        indices = ini_df['index'].values.tolist()
        x = features[indices, :]
        np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved."
        if nfeatures is not None:
            # save trans nonlinear features
            nx = nfeatures[indices, :]
            np.save(path + '{}-{}-features.npy'.format(self.args.graph_lang, self.args.tgtlang), nx)
            print("trans features saved")
            message += "Nonlinear Trans Features saved.\n\n"

        if not self.args.is_static:
            inidays = 1
            j = 0

            for i in range(inidays, len(distinct_dates)):
                print("Start constructing graph ", str(i - j), " ...")
                message += "\nStart constructing graph "
                message += str(i - j)
                message += " ...\n"
                incr_df = df.loc[df['date'] == distinct_dates[i]]
                path = save_path + str(i - j) + '/'
                if not os.path.exists(path):
                    os.mkdir(path)
                np.save(path + "/" + "dataframe.npy", incr_df)

                G = self.construct_graph_from_df(
                    incr_df)  # remove obsolete, version 2: construct graph using only the data of the day
                grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
                message += graph_message
                print("Graph ", str(i - j), " saved")
                message += "Graph "
                message += str(i - j)
                message += " saved\n"
                # record the total number of tweets
                data_split.append(incr_df.shape[0])
                # record the time spent for graph conversion
                all_graph_mins.append(grap_mins)
                # extract and save the labels of corresponding tweets
                # y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
                y = [int(each) for each in incr_df['event_id'].values]
                np.save(path + 'labels.npy', y)
                # incr_df['created_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                np.save(path + "time.npy", incr_df['created_at'].values)
                print("Labels saved.")
                message += "Labels saved.\n"
                # extract and save the features of corresponding tweets
                indices = incr_df['index'].values.tolist()
                x = features[indices, :]
                # x = np.concatenate([x, x_incr], axis = 0)
                np.save(path + 'features.npy', x)
                np.save(path + 'df.npy', incr_df)
                print("Features saved.")
                message += "Features saved."
                if nfeatures is not None:
                    # save trans nonlinear features
                    nx = nfeatures[indices, :]
                    np.save(path + '{}-{}-features.npy'.format(self.args.graph_lang, self.args.tgtlang), nx)
                    print("trans features saved")
                    message += "trans features saved.\n"
        return message, data_split, all_graph_mins



# load Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]


# save graph statistics to save path
def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes


if __name__ == '__main__':
    # from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    args = args_define().args
    
    clkd = CLKD(args)

    clkd.preprocess()

    clkd.fit()  # 训练模型
    predictions, ground_truths = clkd.detection()  # 进行预测
    results = clkd.evaluate(predictions, ground_truths)  # 评估模型
