import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from loader import Corpus, Dictionary

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1345)

# lazy configuration list
fn = './dataset/tinier_lyric.txt'
context_size = 2
learning_rate = 1e-04
niter = 7000
eval_iter = 5
ndim = 32
# when false, it trains to generate text
train_model = False
batch_size = 1024

corpus = Corpus(fn, context_size=context_size, batch_size=batch_size)
vocab_size = len(corpus.dictionary)

print(len(corpus.dictionary))

#TODO: language model:
# sequential - two linears, final softmax


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
        # TODO: train flag, batches, view, loss function, backpropagation, step

        if iter%100 == 0:
            print(f"{iter} train loss: {evaluate_loss('train')} validation loss: {evaluate_loss('val')}")

    torch.save(model.state_dict(), './outputs/bigram.pt')
else:
    #assert first
    assert os.path.exists('./outputs/bigram.pt')
    model.load_state_dict(torch.load('./outputs/bigram.pt'))
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of parameters: {total_params}')

    with torch.no_grad():
        prompt = "I am"
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
