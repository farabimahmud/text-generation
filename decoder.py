import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from attention import SelfAttention
from torchviz import make_dot

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
      self,
      trg_vocab_size,
      embed_size, 
      num_layers, 
      heads, 
      forward_expansion, 
      dropout, 
      device, 
      max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads,
                forward_expansion, dropout, device) 
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(
            x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)

        return out


if __name__== "__main__":
  x = torch.tensor([[1, 5, 6, 4, 5, 6, 4,5, 6, 4,3, 9, 5, 2, 0]])
  trg = torch.tensor([[1, 5, 6, 4, 5, 6, 4,5, 6, 4,3, 9, 5, 2, 5 ,0]])

  # trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])
  src_pad_idx = 0
  trg_pad_idx = 0
  src_vocab_size = 10
  trg_vocab_size = 10
  device = "cpu"
  model = Decoder(10,512,1,4, "cpu",4, 0.2, 200)
  out = model(x, trg[:,:-1])
  print(out.shape)
  # make_dot(out, show_attrs=True, params=dict(model.named_parameters()))
  make_dot(out, params=dict(model.named_parameters())).render("decoder",format="png")
