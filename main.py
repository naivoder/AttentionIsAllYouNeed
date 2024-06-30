import torch


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, h=8, dmodel=512):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.dmodel = dmodel
        self.dk = self.h // self.dmodel
        # self.dq = self.dv = self.dk   # pretty sure the paper says this should be true...

        self.Q = torch.nn.Linear(self.dk, self.dk)
        self.K = torch.nn.Linear(self.dk, self.dk)
        self.V = torch.nn.Linear(self.dk, self.dk)
        self.out = torch.nn.Linear(self.dmodel, self.dmodel)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        energy = torch.dot(Q, K.T) / torch.sqrt(self.dk)
        if mask:
            energy = energy.masked_fill(mask == 0, torch.float("-1e8"))
        return torch.nn.functional.softmax(energy) * V

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        Q = Q.reshape(batch_size, Q.shape[1], self.h, self.dk)
        K = Q.reshape(batch_size, K.shape[1], self.h, self.dk)
        V = V.reshape(batch_size, V.shape[1], self.h, self.dk)

        attention = self.scaled_dot_product_attention(Q, K, V, mask)

        return self.out(attention)
