import torch


class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()


class Encoder(torch.nn.Module):
    def __init__(
        self, vocab_size, max_length, n_layers=6, dmodel=512, h=8, expand=4, dropout=0.1
    ):
        super(Encoder, self).__init__()
        self.input_embedding = torch.nn.Embedding(vocab_size, dmodel)
        self.positional_encoding = torch.nn.Embedding(max_length, dmodel)
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [EncoderBlock(dmodel, h, expand, dropout) for _ in range(n_layers)]
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, mask=None):
        batch_size, input_length = x.shape
        ids = torch.arange(0, input_length).expand(batch_size, input_length)
        ids = self.positional_encoding(ids.to(self.device))
        x = self.dropout(self.input_embedding(x) + ids)
        for transformer in self.layers:
            # special case where q, k, v are all the same
            x = transformer(query=x, key=x, value=x, mask=mask)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, dmodel=512, h=8, expand=4, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(dmodel, h)
        self.feed_forward = FeedForwardNetwork(dmodel, expand)
        self.ln1 = torch.nn.LayerNorm(dmodel)
        self.ln2 = torch.nn.LayerNorm(dmodel)
        self.dropout = torch.nn.Dropout(dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, query, key, value, mask=None):
        x = self.attention(query, key, value, mask)
        x = self.dropout(self.ln1(x + query))
        y = self.feed_forward(x)
        return self.dropout(self.ln2(x + y))


class Decoder(torch.nn.Module):
    def __init__(
        self, vocab_size, max_length, n_layers=6, dmodel=512, h=8, expand=4, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.output_embedding = torch.nn.Embedding(vocab_size, dmodel)
        self.positional_encoding = torch.nn.Embedding(max_length, dmodel)
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [DecoderBlock(dmodel, h, expand, dropout) for _ in range(n_layers)]
        )
        self.out = torch.nn.Linear(dmodel, vocab_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, src, tgt, mask, src_mask=None):
        batch_size, input_length = src.shape
        ids = torch.arange(0, input_length).expand(batch_size, input_length)
        ids = self.positional_encoding(ids.to(self.device))
        x = self.dropout(self.input_embedding(tgt) + ids)
        for decoder_block in self.layers:
            x = decoder_block(query=x, key=src, value=src, mask=mask, src_mask=src_mask)
        return self.out(x)


class DecoderBlock(torch.nn.Module):
    def __init__(self, dmodel=512, h=8, expand=4, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadSelfAttention(dmodel, h)
        self.transformer = EncoderBlock(dmodel, h, expand, dropout)
        self.ln = torch.nn.LayerNorm(dmodel)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, key, value, mask, src_mask=None):
        masked = self.masked_attention(x, x, x, mask)
        query = self.dropout(self.ln(masked + x))
        # can use src_mask to avoid computation on padded inputs
        return self.transformer(query, key, value, src_mask)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, dmodel=512, expand=4):
        super(FeedForwardNetwork, self).__init__()
        self.h1 = torch.nn.Linear(dmodel, dmodel * expand)
        self.out = torch.nn.Linear(dmodel * expand, dmodel)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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
