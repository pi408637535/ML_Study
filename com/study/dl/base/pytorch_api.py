import torch.nn as nn
import torch as t

if __name__ == '__main__':
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    src = t.rand(10, 32, 512)
    out = encoder_layer(src)
    print(out.shape)