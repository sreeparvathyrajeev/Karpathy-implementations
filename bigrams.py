#i have commented out certain portions which were part of the learning process but are not required while running the code
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F

words= open('names.txt', 'r').read().splitlines()
#b={}
#for w in words:
#    chs=['<S>']+list(w)+['<E>']
#    for ch1, ch2 in zip(chs,chs[1:]):
#        bigram=(ch1,ch2)
#       b[bigram]=b.get(bigram,0)+1
#b=sorted(b.items(),key= lambda kv: -kv[1])
#print(b[:10])

N= torch.zeros((27,27), dtype=torch.int32)
#print(N)

chars=sorted(list(set(''.join(words))))
#print(chars)

stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
#print(itos)

for w in words:
    chs=['.']+list(w)+['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2]+=1
#print(N)

#plt.imshow(N.numpy())
#plt.show()


P= N.float()
P=P/P.sum(0,keepdim=True)

g= torch.Generator().manual_seed(2147483647)
for i in range(10):
    ix=0
    out=[]
    while True:
        p= P[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    # print(''.join(out))


log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

# print(f'{log_likelihood=}')
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')







#creating the training set of bigrams
xs,ys=[],[]
for w in words:
    chs = ['.']+list(w)+['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs=torch.tensor(xs)
ys=torch.tensor(ys)
num=xs.nelement()
# print(xs)
# print(ys)



g= torch.Generator().manual_seed(2147483647)
W= torch.randn((27,27),generator=g, requires_grad=True)




# nlls=torch.zeros(5)
# for i in range(5):
#     x=xs[i].item()
#     y=ys[i].item()
#     print("bigram:", itos[x], itos[y], "index:", x, y)
#     print("probabilties of each possible next character:", probs[i])
#     print("probability of actual next character:", probs[i,y].item())
#     nll=-torch.log(probs[i,y])
#     print("negative log likelihood:", nll.item())
#     nlls[i]=nll
# print("avg nll(loss):", nlls.mean().item())


for i in range(100):
    #forward pass
    xenc=F.one_hot(xs,num_classes=27).float()
    logits= xenc @ W
    counts= logits.exp()
    probs= counts/counts.sum(1,keepdim=True)
    loss=-probs[torch.arange(num),ys].log().mean()
    print(f"iteration {i}, loss: {loss.item()}")

    #backward pass
    W.grad=None #clear the gradient
    loss.backward()
    
    #update the weights using gradient descent
    W.data+= -50*W.grad

#sampling from the neural net
g= torch.Generator().manual_seed(2147483647)
for i in range(10):
    ix=0
    out=[]
    while True:
        xenc=F.one_hot(torch.tensor([ix]),num_classes=27).float()
        logits= xenc @ W
        counts = logits.exp()
        p= counts/counts.sum(1,keepdim=True)    
        
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))

