import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # unnormalized attention weights    
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

# How many GPUs are there?
print(torch.cuda.device_count())

vocab_size = 50_000
torch.manual_seed(123)

sentence = 'Hello world'

dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)

embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

print("X")
# print(embedded_sentence)
# print(embedded_sentence.shape)
print(f"X.shape={embedded_sentence.shape}")
print(torch.flatten(embedded_sentence))

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 2, 2, 4

W_query = torch.nn.Parameter(torch.rand(d, d_q))
# print(W_query)
print(f"W_query.shape={W_query.shape}")
print(torch.flatten(W_query))
W_key = torch.nn.Parameter(torch.rand(d, d_k))
# print(W_key)
print(f"W_key.shape={W_key.shape}")
print(torch.flatten(W_key))
W_value = torch.nn.Parameter(torch.rand(d, d_v))
# print(W_value)
print(f"W_value.shape={W_value.shape}")
print(torch.flatten(W_value))

d_in, d_out_kq, d_out_v = d, 2, 4

sa = SelfAttention(d_in, d_out_kq, d_out_v)
print("Self-attention output:")
print(sa(embedded_sentence).shape)
print(torch.flatten(sa(embedded_sentence)))