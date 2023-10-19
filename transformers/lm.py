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

#TODO: feedforward, sequential: lin*4, relu, lin, dropout

#TODO: Attention Head: k, q, v, k.t(-2,-1), mask, sm, dropout, @

#TODO: SelfAttention: heads, lin, dropout

#TODO: Decoderlayer: x + at(ln(x)), x + ffwd(ln(x))

#TODO: LM: embeddings (arange), decoders, ln, fd1, rel, fd2, fm

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
        pass
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
