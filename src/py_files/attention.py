import torch

# How many GPUs are there?
print(torch.cuda.device_count())

# Get the name of the current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Is PyTorch using a GPU?
print(torch.cuda.is_available())

vocab_size = 50_000
torch.manual_seed(123)

sentence = 'Life is short, eat dessert first'

dc = {s:i for i,s 
      in enumerate(sorted(sentence.replace(',', '').split()))}

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', '').split()]
)

embed = torch.nn.Embedding(vocab_size, 3)
embedded_sentence = embed(sentence_int).detach()

print("X")
print(embedded_sentence)
print(f"X.shape={embedded_sentence.shape}")
print(torch.flatten(embedded_sentence))