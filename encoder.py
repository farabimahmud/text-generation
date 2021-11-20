from transformer_block import TransformerBlock
import torch.nn as nn
import torch
from torchviz import make_dot


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(
            N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

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
  model = Encoder(10,512,1,4, "cpu",4, 0.2, 200)
  out = model(x, trg[:,:-1])
  print(out.shape)
  # make_dot(out, show_attrs=True, params=dict(model.named_parameters()))
  make_dot(out, params=dict(model.named_parameters())).render("encoder",format="png")
