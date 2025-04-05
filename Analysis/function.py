import os
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from contextlib import nullcontext
import types

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
#from cusomized_gradio_blocks import create_myexamples, customized_as_example, customized_postprocess

import numpy as np
import torch
import torchvision.transforms as tvtrans
from PIL import Image
from IPython.display import display
from lib.model_zoo import get_model


def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')


def remove_duplicate_word(tx):
    def combine_words(input, length):
        combined_inputs = []
        if len(splitted_input)>1:
            for i in range(len(input)-1):
                combined_inputs.append(input[i]+" "+last_word_of(splitted_input[i+1],length)) #add the last word of the right-neighbour (overlapping) sequence (before it has expanded), which is the next word in the original sentence
        return combined_inputs, length+1

    def remove_duplicates(input, length):
        bool_broke=False #this means we didn't find any duplicates here
        for i in range(len(input) - length):
            if input[i]==input[i + length]: #found a duplicate piece of sentence!
                for j in range(0, length): #remove the overlapping sequences in reverse order
                    del input[i + length - j]
                bool_broke = True
                break #break the for loop as the loop length does not matches the length of splitted_input anymore as we removed elements
        if bool_broke:
            return remove_duplicates(input, length) #if we found a duplicate, look for another duplicate of the same length
        return input

    def last_word_of(input, length):
        splitted = input.split(" ")
        if len(splitted)==0:
            return input
        else:
            return splitted[length-1]

    def split_and_puncsplit(text):
        tx = text.split(" ")
        txnew = []
        for txi in tx:
            txqueue=[]
            while True:
                if txi[0] in '([{':
                    txqueue.extend([txi[:1], '<puncnext>'])
                    txi = txi[1:]
                    if len(txi) == 0:
                        break
                else:
                    break
            txnew += txqueue
            txstack=[]
            if len(txi) == 0:
                continue
            while True:
                if txi[-1] in '?!.,:;}])':
                    txstack = ['<puncnext>', txi[-1:]] + txstack
                    txi = txi[:-1]
                    if len(txi) == 0:
                        break
                else:
                    break
            if len(txi) != 0:
                txnew += [txi]
            txnew += txstack
        return txnew

    if tx == '':
        return tx

    splitted_input = split_and_puncsplit(tx)
    word_length = 1
    intermediate_output = False
    while len(splitted_input)>1:
        splitted_input = remove_duplicates(splitted_input, word_length)
        if len(splitted_input)>1:
            splitted_input, word_length = combine_words(splitted_input, word_length)
        if intermediate_output:
            print(splitted_input)
            print(word_length)
    output = splitted_input[0]
    output = output.replace(' <puncnext> ', '')
    return output


def decompose(x, q=20, niter=100):
    x_mean = x.mean(-1, keepdim=True)
    x_input = x - x_mean
    u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
    ss = torch.stack([torch.diag(si) for si in s])
    x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
    x_remain = x_input - x_lowrank
    return u, s, v, x_mean, x_remain


class adjust_rank(object):
    def __init__(self, max_drop_rank=[1, 5], q=20):
        self.max_semantic_drop_rank = max_drop_rank[0]
        self.max_style_drop_rank = max_drop_rank[1]
        self.q = q

        def t2y0_semf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_semf = t2y0_semf_wrapper(t0, y00, t1, y01)

        def x2y_semf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = 0
        x1, y1 = self.max_semantic_drop_rank+1, 1
        self.x2y_semf = x2y_semf_wrapper(x0, x1, y1)
        
        def t2y0_styf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_styf = t2y0_styf_wrapper(t0, y00, t1, y01)

        def x2y_styf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = q-1
        x1, y1 = self.max_style_drop_rank-1, 1
        self.x2y_styf = x2y_styf_wrapper(x0, x1, y1)

    def __call__(self, x, lvl):
        if lvl == 0.5:
            return x

        if x.dtype == torch.float16:
            fp16 = True
            x = x.float()
        else:
            fp16 = False
        std_save = x.std(axis=[-2, -1])

        u, s, v, x_mean, x_remain = decompose(x, q=self.q)

        if lvl < 0.5:
            assert lvl>=0
            for xi in range(0, self.max_semantic_drop_rank+1):
                y0 = self.t2y0_semf(lvl)
                yi = self.x2y_semf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi

        elif lvl > 0.5:
            assert lvl <= 1
            for xi in range(self.max_style_drop_rank, self.q):
                y0 = self.t2y0_styf(lvl)
                yi = self.x2y_styf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi
            x_remain = 0

        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_new = x_lowrank + x_mean + x_remain

        std_new = x_new.std(axis=[-2, -1])
        x_new = x_new / std_new * std_save

        if fp16:
            x_new = x_new.half()

        return x_new



































