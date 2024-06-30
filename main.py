import torch


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, h=8, dmod=512):
        super(MultiHeadSelfAttention, self).__init__()
        self.h = h
        self.dmod = dmod
        self.dk = self.h // self.dmod

        self.Q_projections = [torch.nn.Linear(self.dk, self.dk) for _ in range(self.h)]
        self.K_projections = [torch.nn.Linear(self.dk, self.dk) for _ in range(self.h)]
        self.V_projections = [torch.nn.Linear(self.dk, self.dk) for _ in range(self.h)]

    def scaled_dot_product_attention(self, Q, K, V):
        scaled_dot_prod = torch.dot(Q, K.T) / torch.sqrt(self.dk)
        return torch.nn.functional.softmax(scaled_dot_prod) * V

    def forward(self, Q, K, V):
        heads = []
        for i in range(self.h):
            Q = self.Q_projections[i](Q)
            K = self.K_projections[i](K)
            V = self.V_projections[i](V)
            heads.append(self.scaled_dot_product_attention(Q, K, V))
        heads = torch.cat(heads)
