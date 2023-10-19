import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from loader import Corpus, Dictionary

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1345)

# lazy configuration list
fn = './dataset/tiny_lyric.txt'
context_size = 32
learning_rate = 1e-04
niter = 10000
eval_iter = 5
ndim = 64
# when false, it trains to generate text
train_model = False
batch_size = 128
nlayers = 6
nheads = 4
pdropout = 0.2

corpus = Corpus(fn, context_size=context_size, batch_size=batch_size)
vocab_size = len(corpus.dictionary)

print(len(corpus.dictionary))

class Feedforward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ndim, ndim * 4),
            nn.ReLU(),
            nn.Linear(ndim * 4, ndim),
            nn.Dropout(pdropout)
        )

    def forward(self, x):
        x = self.net(x)
        # perhaps dropout
        return x


class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(ndim, head_size, bias=False)
        self.query = nn.Linear(ndim, head_size, bias=False)
        self.value = nn.Linear(ndim, head_size, bias=False)
        self.dropout = nn.Dropout(pdropout)

    def forward(self, x):
        k = self.key(x); q = self.query(x); v = self.value(x)
        p = (q@k.transpose(-2, -1))/self.head_size**-0.5
        #apply masking 
        _tril = torch.tril(torch.ones(context_size, context_size)).to(device)
        _tril = _tril.to(device)
        p = p.masked_fill(_tril==0, float('-inf'))
        p = F.softmax(p, dim=-1)
        # dropout at p
        p = self.dropout(p)
        hattn = p @ v
        return hattn


class SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(ndim//nheads) for _ in range(nheads)])
        self.ffwd = nn.Linear(ndim, ndim)
        self.dropout = nn.Dropout(pdropout)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.ffwd(x)
        out = self.dropout(out)
        return out

class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = SelfAttention()
        self.layernorm1 = nn.LayerNorm(ndim)
        self.ffwd = Feedforward()
        self.layernorm2 = nn.LayerNorm(ndim)

    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ffwd(self.layernorm2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, ndim)
        self.positional_embedding = nn.Embedding(context_size, ndim)
        self.decoders = nn.Sequential(
            *[DecoderLayer() for _ in range(nlayers)]
        )
        self.layernorm = nn.LayerNorm(ndim)
        self.ffwd1 = nn.Linear(ndim, ndim)
        self.relu = nn.ReLU()
        self.ffwd2 = nn.Linear(ndim, vocab_size)


    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding(torch.arange(context_size).to(device))
        x = self.decoders(x)
        x = self.layernorm(x)        
        x = self.ffwd1(x)
        x = self.relu(x)
        x = self.ffwd2(x)
        x = F.log_softmax(x, dim=-1)
        return x


@torch.no_grad()
def evaluate_loss(_stage):
    model.eval()
    total_loss = 0.0
    for _ in range(eval_iter):
        X, y = corpus.get_batch(_stage)
        out = model(X)
        B, T, C = out.shape
        out = out.view(B*T, C)
        y = y.view(B*T)
        loss = loss_fn(out, y.flatten())
        total_loss += loss.item()
    return total_loss / eval_iter


model = LanguageModel().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.NLLLoss()

# training
if train_model:
    for iter in range(niter):
        model.train()
        X, y = corpus.get_batch('train')
        out = model(X)
        B, T, C = out.shape
        out = out.view(B*T, C)
        y = y.view(B*T)
        loss = loss_fn(out, y.flatten())
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        if iter%100 == 0:
            print(f"{iter} train loss: {evaluate_loss('train')} validation loss: {evaluate_loss('val')}")
        
        if iter % 1000 == 0:
            torch.save(model.state_dict(), f'./outputs/lm{iter}.pt')

    torch.save(model.state_dict(), './outputs/lm.pt')
else:
    #assert first
    assert os.path.exists('./outputs/lm.pt')
    model.load_state_dict(torch.load('./outputs/lm.pt'))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of parameters: {total_params}')

    with torch.no_grad():
        prompt = "I am lucky to love you baby I am lucky to have known you I am lucky to have you I am lucky to have seen these days baby I will always"
        tokens = corpus.dictionary.encode(prompt.split()[:context_size])
        generate_length = 20
        for _ in range(generate_length):
            _input = torch.tensor(tokens[len(tokens)-context_size:], dtype=torch.long)
            _input = _input.view(1,context_size).to(device)
            out = model(_input)
            B, T, C = out.shape # expecting B to be 1
            out = out.view(T, C)
            out = F.softmax(out, dim=-1)
            print(f'out shape {out.shape}')
            prev_token = torch.multinomial(out[0,:], 1)
            print(f'{corpus.dictionary.decode([prev_token.item()])}')
            nxt_token = torch.multinomial(out[-1,:], 1)
            tokens.append(nxt_token.item())
        print(corpus.dictionary.decode(tokens))
