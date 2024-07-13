import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


class Transformer(torch.nn.Module):
    def __init__(self, dmodel=512, h=8, expand=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.attention = MultiHeadSelfAttention(dmodel, h)
        self.feed_forward = FeedForwardNetwork(dmodel, expand)
        self.ln1 = torch.nn.LayerNorm(dmodel)
        self.ln2 = torch.nn.LayerNorm(dmodel)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        x = self.attention(query, key, value, mask)
        x = self.dropout(self.ln1(x + query))
        y = self.feed_forward(x)
        return self.dropout(self.ln2(x + y))


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, dmodel=512, expand=4):
        super(FeedForwardNetwork, self).__init__()
        self.h1 = torch.nn.Linear(dmodel, dmodel * expand)
        self.out = torch.nn.Linear(dmodel * expand, dmodel)

    def forward(self, x):
        x = torch.nn.functional.relu(self.h1(x))
        return self.out(x)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, dmodel=512, h=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.dmodel = dmodel
        self.dk = self.dmodel // self.h

        assert self.dmodel % self.h == 0

        self.wq = torch.nn.Linear(self.dk, self.dk, bias=False)
        self.wk = torch.nn.Linear(self.dk, self.dk, bias=False)
        self.wv = torch.nn.Linear(self.dk, self.dk, bias=False)

        self.out = torch.nn.Linear(self.dmodel, self.dmodel)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.h, self.dk)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]

        queries = self.split_heads(self.wq(queries), batch_size)
        keys = self.split_heads(self.wk(keys), batch_size)
        values = self.split_heads(self.wv(values), batch_size)

        # https://youtu.be/U0s0f995w14?si=KXtErolSQ-w6OHoX
        # queries: (batch_size, query_len, h, dk)
        # keys: (batch_size, key_len, h, dk)
        # energy: (batch_size, h, query_len, key_len)
        energy = torch.einsum("abcd,aecd->acbe", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        energy /= torch.sqrt(torch.FloatTensor(self.dk))

        attention = torch.nn.functional.softmax(energy, dim=-1)

        # note: key_len and value_len are always equal!
        # attention: (batch_size, h, query_len, key_len)
        # values: (batch_size, value_len, h, dk)
        # context: (batch_size, query_len, h, dk)
        context = torch.einsum("abcd,adbf->acbf", [attention, values])
        context = context.view(batch_size, -1, self.d_model)

        return self.out(context)
