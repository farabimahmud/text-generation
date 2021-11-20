import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder, DecoderBlock
from torchviz import make_dot

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 heads=8,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 dropout=0,
                 device="cpu",
                 max_len=100
                 ):

        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers,
                               heads, device, forward_expansion, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers,
                               heads, forward_expansion, dropout, device, max_len)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device


    def make_src_mask(self, src):
      src_mask = (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
      return src_mask.to(self.device)

    def make_trg_mask(self, trg):
      N, trg_len  = trg.shape 
      trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len,trg_len
      )
      return trg_mask.to(self.device)

    def forward(self, src, trg):
      src_mask = self.make_src_mask(src)
      trg_mask = self.make_trg_mask(trg)
      enc_src = self.encoder(src,src_mask)
      out = self.decoder(trg, enc_src, src_mask, trg_mask)
      return out 


if __name__== "__main__":
  x = torch.tensor([[1, 5, 6, 4, 5, 6, 4,5, 6, 4,3, 9, 5, 2, 0], [1, 5, 6, 4, 5, 6, 4,5, 6, 4,3, 9, 5, 2, 0]])
  trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])
  src_pad_idx = 0
  trg_pad_idx = 0
  src_vocab_size = 10
  trg_vocab_size = 10
  device = "cpu"
  model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,device=device)
  out = model(x, trg[:,:-1])
  print(out.shape)
  # make_dot(out, show_attrs=True, params=dict(model.named_parameters()))
  make_dot(out, params=dict(model.named_parameters())).render(format="png")
