# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 16:20
# @Author  : piguanghua
# @FileName: pytorch_crf.py
# @Software: PyCharm

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch as t
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchtext
import random
import copy


torch.manual_seed(1)

#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the f_forward_algorward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    #feats: seq,tag
    def _forward_alg(self, feats):

        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        previous = init_alphas

        for feat in feats: #feats:sequence,tag
            word_tag_score = [] # 计算逐个word->tag score
            for tag in range(self.tagset_size):
                word_emission = feat[tag].expand(1, self.tagset_size)
                word_tag_transition = self.transitions[tag, :]
                word_score = previous + word_emission + word_tag_transition
                word_score = log_sum_exp(word_score)

                word_tag_score.append(word_score.view(1))

            previous = t.cat(word_tag_score, dim=0).view(1, -1)

        terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
        all_path_score = log_sum_exp(terminal_var)
        return all_path_score


    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) #lstm_out:batch,seq,hidden*dir
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out) #seq,tag
        return lstm_feats

    def _score_sentence(self, feats, tags):
        #feats: seq,tag
        #tags:tag
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        #tags[0] is start tag.
        for i,feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]]  + feat[tags[i + 1]]

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score


    #feats seq,tags
    def _viterbi_decode(self, feats):
        seq_length, tag_size = feats.size()
        f = torch.zeros(seq_length, tag_size)


        for step in range(1, len(seq_length)):
            for j in range(self.tag_size):
                f[step,j] =




        pass

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# Run training


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


######################################################################
# Exercise: A new loss function for discriminative tagging
# --------------------------------------------------------
#
# It wasn't really necessary for us to create a computation graph when
# doing decoding, since we do not backpropagate from the viterbi path
# score. Since we have it anyway, try training the tagger where the loss
# function is the difference between the Viterbi path score and the score
# of the gold-standard path. It should be clear that this function is
# non-negative and 0 when the predicted tag sequence is the correct tag
# sequence. This is essentially *structured perceptron*.
#
# This modification should be short, since Viterbi and score\_sentence are
# already implemented. This is an example of the shape of the computation
# graph *depending on the training instance*. Although I haven't tried
# implementing this in a static toolkit, I imagine that it is possible but
# much less straightforward.
#
# Pick up some real data and do a comparison!
#