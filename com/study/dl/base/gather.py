import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


if __name__ == '__main__':
    data = t.randn((1,4,5))
    index = t.randint(high=4,size=(1,4,2))
    result = t.gather(data, dim= 2, index= index)
    print(result)

    data = t.randn((1, 4, 5))
    index = t.randint(high=4, size=(1, 2, 5))
    result = t.gather(data, dim=1, index=index)
    print(result)