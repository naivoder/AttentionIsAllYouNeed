import torch


class Transformer(torch.nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    Vaswani, Ashish, et al. 'Attention is all you need.' https://arxiv.org/abs/1706.03762

    Parameters
    ----------
    src_vocab_size : int
        Size of the source vocabulary.
    tgt_vocab_size : int
        Size of the target vocabulary.
    pad_token : int, optional
        Token used for padding sequences (default is 0).
    dmodel : int, optional
        Dimension of the model (default is 512).
    max_length : int, optional
        Maximum length of input sequences (default is 100).
    n_layers : int, optional
        Number of layers in the encoder and decoder (default is 6).
    h : int, optional
        Number of attention heads (default is 8).
    expand : int, optional
        Expansion factor in the feed-forward network (default is 4).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        pad_token=-1,
        dmodel=512,
        max_length=100,
        n_layers=6,
        h=8,
        expand=4,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.pad_token = pad_token

        self.encoder = Encoder(
            src_vocab_size, max_length, n_layers, dmodel, h, expand, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, max_length, n_layers, dmodel, h, expand, dropout
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _get_src_mask(self, src):
        """
        Generate source mask for padding tokens.

        Parameters
        ----------
        src : torch.Tensor
            Source sequence tensor.

        Returns
        -------
        torch.Tensor
            Source mask tensor.
        """
        src_mask = (src != self.pad_token)[:, None, None, :]
        return src_mask.to(self.device)

    def _get_tgt_mask(self, tgt):
        """
        Generate target mask for masking future tokens.

        Parameters
        ----------
        tgt : torch.Tensor
            Target sequence tensor.

        Returns
        -------
        torch.Tensor
            Target mask tensor.
        """
        batch_size, tgt_length = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_length, tgt_length)))
        tgt_mask = tgt_mask.expand(batch_size, 1, tgt_length, tgt_length)
        return tgt_mask.to(self.device)

    def forward(self, src, tgt):
        """
        Forward pass for the Transformer model.

        Parameters
        ----------
        src : torch.Tensor
            Source sequence tensor.
        tgt : torch.Tensor
            Target sequence tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying softmax.
        """
        src_mask = self._get_src_mask(src)
        tgt_mask = self._get_tgt_mask(tgt)
        x = self.encoder(src, src_mask)
        x = self.decoder(x, tgt, tgt_mask, src_mask)
        return torch.nn.functional.softmax(x, -1)


class Encoder(torch.nn.Module):
    """
    Encoder module for the Transformer model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    max_length : int
        Maximum length of input sequences.
    n_layers : int, optional
        Number of layers in the encoder (default is 6).
    dmodel : int, optional
        Dimension of the model (default is 512).
    h : int, optional
        Number of attention heads (default is 8).
    expand : int, optional
        Expansion factor in the feed-forward network (default is 4).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

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
        """
        Forward pass for the Encoder module.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor.
        mask : torch.Tensor, optional
            Mask tensor for padding tokens (default is None).

        Returns
        -------
        torch.Tensor
            Encoded output tensor.
        """
        batch_size, input_length = x.shape
        ids = torch.arange(0, input_length).expand(batch_size, input_length)
        ids = self.positional_encoding(ids.to(self.device))
        x = self.dropout(self.input_embedding(x) + ids)
        for transformer in self.layers:
            # special case where q, k, v are all the same
            x = transformer(query=x, key=x, value=x, mask=mask)
        return x


class EncoderBlock(torch.nn.Module):
    """
    Encoder block for the Transformer model.

    Parameters
    ----------
    dmodel : int, optional
        Dimension of the model (default is 512).
    h : int, optional
        Number of attention heads (default is 8).
    expand : int, optional
        Expansion factor in the feed-forward network (default is 4).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

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
        """
        Forward pass for the Encoder block.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor.
        key : torch.Tensor
            Key tensor.
        value : torch.Tensor
            Value tensor.
        mask : torch.Tensor, optional
            Mask tensor (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor after applying attention and feed-forward network.
        """
        x = self.attention(query, key, value, mask)
        x = self.dropout(self.ln1(x + query))
        y = self.feed_forward(x)
        return self.dropout(self.ln2(x + y))


class Decoder(torch.nn.Module):
    """
    Decoder module for the Transformer model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    max_length : int
        Maximum length of target sequences.
    n_layers : int, optional
        Number of layers in the decoder (default is 6).
    dmodel : int, optional
        Dimension of the model (default is 512).
    h : int, optional
        Number of attention heads (default is 8).
    expand : int, optional
        Expansion factor in the feed-forward network (default is 4).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

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
        """
        Forward pass for the Decoder module.

        Parameters
        ----------
        src : torch.Tensor
            Encoded source tensor.
        tgt : torch.Tensor
            Target sequence tensor.
        mask : torch.Tensor
            Mask tensor for target sequence.
        src_mask : torch.Tensor, optional
            Mask tensor for source sequence (default is None).

        Returns
        -------
        torch.Tensor
            Decoded output tensor.
        """
        batch_size, tgt_length = tgt.shape
        ids = torch.arange(0, tgt_length).expand(batch_size, tgt_length)
        ids = self.positional_encoding(ids.to(self.device))
        x = self.dropout(self.output_embedding(tgt) + ids)
        for decoder_block in self.layers:
            x = decoder_block(x, key=src, value=src, mask=mask, src_mask=src_mask)
        return self.out(x)


class DecoderBlock(torch.nn.Module):
    """
    Decoder block for the Transformer model.

    Parameters
    ----------
    dmodel : int, optional
        Dimension of the model (default is 512).
    h : int, optional
        Number of attention heads (default is 8).
    expand : int, optional
        Expansion factor in the feed-forward network (default is 4).
    dropout : float, optional
        Dropout rate (default is 0.1).
    """

    def __init__(self, dmodel=512, h=8, expand=4, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadSelfAttention(dmodel, h)
        self.transformer = EncoderBlock(dmodel, h, expand, dropout)
        self.ln = torch.nn.LayerNorm(dmodel)
        self.dropout = torch.nn.Dropout(dropout)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, key, value, mask, src_mask=None):
        """
        Forward pass for the Decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Target sequence tensor.
        key : torch.Tensor
            Encoded source tensor.
        value : torch.Tensor
            Encoded source tensor.
        mask : torch.Tensor
            Mask tensor for target sequence.
        src_mask : torch.Tensor, optional
            Mask tensor for source sequence (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor after applying attention and feed-forward network.
        """
        masked = self.masked_attention(x, x, x, mask)
        query = self.dropout(self.ln(masked + x))
        # can use src_mask to avoid computation on padded inputs
        return self.transformer(query, key, value, src_mask)


class FeedForwardNetwork(torch.nn.Module):
    """
    Feed-forward network for the Transformer model.

    Parameters
    ----------
    dmodel : int, optional
        Dimension of the model (default is 512).
    expand : int, optional
        Expansion factor (default is 4).
    """

    def __init__(self, dmodel=512, expand=4):
        super(FeedForwardNetwork, self).__init__()
        self.h1 = torch.nn.Linear(dmodel, dmodel * expand)
        self.out = torch.nn.Linear(dmodel * expand, dmodel)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying feed-forward network.
        """
        x = torch.nn.functional.relu(self.h1(x))
        return self.out(x)


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self-attention mechanism.

    Parameters
    ----------
    dmodel : int, optional
        Dimension of the model (default is 512).
    h : int, optional
        Number of attention heads (default is 8).
    """

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
        """
        Split the input tensor into multiple heads.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        batch_size : int
            Batch size.

        Returns
        -------
        torch.Tensor
            Tensor split into multiple heads.
        """
        return x.reshape(batch_size, -1, self.h, self.dk)

    def forward(self, queries, keys, values, mask=None):
        """
        Forward pass for the multi-head self-attention mechanism.

        Parameters
        ----------
        queries : torch.Tensor
            Query tensor.
        keys : torch.Tensor
            Key tensor.
        values : torch.Tensor
            Value tensor.
        mask : torch.Tensor, optional
            Mask tensor (default is None).

        Returns
        -------
        torch.Tensor
            Output tensor after applying multi-head self-attention.
        """
        batch_size = queries.shape[0]

        queries = self.wq(self.split_heads(queries, batch_size))
        keys = self.wk(self.split_heads(keys, batch_size))
        values = self.wv(self.split_heads(values, batch_size))

        # https://youtu.be/U0s0f995w14?si=KXtErolSQ-w6OHoX
        # queries: (batch_size, query_len, h, dk)
        # keys: (batch_size, key_len, h, dk)
        # -> energy: (batch_size, h, query_len, key_len)
        energy = torch.einsum("abcd,aecd->acbe", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        energy /= torch.sqrt(torch.tensor(self.dk))

        attention = torch.nn.functional.softmax(energy, dim=-1)

        # note: key_len and value_len are always equal!
        # attention: (batch_size, h, query_len, key_len)
        # values: (batch_size, value_len, h, dk)
        # -> context: (batch_size, query_len, h, dk)
        context = torch.einsum("abcd,adbf->acbf", [attention, values])
        context = context.reshape(batch_size, -1, self.dmodel)

        return self.out(context)


if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_sequence_length = 10
    tgt_sequence_length = 8
    src_vocab_size = tgt_vocab_size = 10

    src = torch.tensor(np.random.randint(0, src_vocab_size, (2, src_sequence_length)))
    tgt = torch.tensor(np.random.randint(0, tgt_vocab_size, (2, tgt_sequence_length)))

    model = Transformer(src_vocab_size, tgt_vocab_size)
    print(torch.argmax(model(src.to(device), tgt[:, :-1].to(device)), -1))
