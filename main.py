import torch


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, dmodel=512, h=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.dmodel = dmodel
        self.dk = self.dmodel // self.h

        assert self.dmodel % self.h == 0

        self.wq = torch.nn.Linear(self.dmodel, self.dmodel)
        self.wk = torch.nn.Linear(self.dmodel, self.dmodel)
        self.wv = torch.nn.Linear(self.dmodel, self.dmodel)

        self.out = torch.nn.Linear(self.dmodel, self.dmodel)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.h, self.dk)
        return x.permute(0, 2, 1, 3)

    def forward(self, Q, K, V):
        batch_size = Q.shape[0]

        Q = self.split_heads(self.wq(Q), batch_size)
        K = self.split_heads(self.wk(K), batch_size)
        V = self.split_heads(self.wv(V), batch_size)

        energy = torch.matmul(Q, K.transpose(-2, -1))
        ender /= torch.sqrt(torch.FloatTensor(self.dk))
        attention = torch.nn.functional.softmax(energy, dim=-1)

        context = torch.matmul(attention, V)
        context = context.permute(0, 2, 1, 3)
        context = context.view(batch_size, -1, self.d_model)

        return self.out(context)
