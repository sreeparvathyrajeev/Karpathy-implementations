import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device='cuda'
print(torch.cuda.is_available())  # should print True
print(torch.cuda.get_device_name(0))  # should show your GPU name   

#training set, validation set, test set loaded from file
#80%, 10%, 10%
data = torch.load('dataset.pt') 
Xtr, Ytr = data['Xtr'].to(device), data['Ytr'].to(device)
Xdev, Ydev = data['Xdev'].to(device), data['Ydev'].to(device)
Xte, Yte = data['Xte'].to(device), data['Yte'].to(device)       

vocab_size=27
block_size=3

def cmp(s,dt,t):
    ex=torch.all(dt==t.grad).item()
    app=torch.allclose(dt,t.grad)
    maxdiff=(dt-t.grad).abs().max().item()
    print(f'{s:15s}: exact={str(ex):5s} approx={str(app):5s} maxdiff={maxdiff}')


#model definition

n_embed= 10 #dimensionality of character embedding vector
n_hidden= 64 #number of neurons in hidden layer of mlp

g= torch.Generator().manual_seed(2147483647) #for reproducibility
C= torch.randn((vocab_size,n_embed), generator=g).to(device) #character embedding matrix

#layer 1
W1= (torch.randn((n_embed*block_size,n_hidden), generator=g) * (5/3) /((n_embed*block_size)**0.5)).to(device) #kaiming initialisation for tanh activation function
b1= (torch.randn(n_hidden, generator=g) * 0.1).to(device) #biases can be initialized to small random values or zeros, as they will be learned during training and do not require special initialization like weights. Initializing biases to small random values can help break symmetry and allow the model to learn more effectively, while initializing them to zeros is also a common practice and can work well in many cases. The choice between these two options may depend on the specific architecture of the model and the preferences of the practitioner.

#layer 2
W2= (torch.randn((n_hidden,vocab_size), generator=g) * 0.1).to(device)
b2= (torch.randn(vocab_size, generator=g) * 0.1).to(device)

#batchnorm parameters
bngain= (torch.randn((1,n_hidden), generator=g) * 0.1 + 1.0).to(device)
bnbias= (torch.randn((1,n_hidden), generator=g) * 0.1).to(device)

parameters= [C,W1,b1,W2,b2,bngain,bnbias]
for p in parameters:
    p.requires_grad = True

print(sum(p.nelement() for p in parameters)) #number of parameters in the model


#construct minibatch
batch_size=32
n=batch_size
ix= torch.randint(0,Xtr.shape[0],(batch_size,), generator=g)
Xb, Yb= Xtr[ix], Ytr[ix]


#forward pass that is chunkated into steps for clarity when we do backpropagation by hand
#embedding layer
emb= C[Xb] # (batch_size,block_size,n_embed)
print(f'emb.shape: {emb.shape}')
embcat= emb.view(emb.shape[0], -1) # (batch_size, block_size*n_embed)
print(f'embcat.shape: {embcat.shape}')
#linear layer 1
hprebn= embcat @ W1 +b1 # (batch_size, n_hidden) and b1 is broadcasted to (batch_size, n_hidden)
print(f'hprebn.shape: {hprebn.shape}')
#batchnorm layer
bnmeani= 1/n * hprebn.sum(dim=0, keepdim=True) # (1, n_hidden)
print(f'bnmeani.shape: {bnmeani.shape}')
bndiff= hprebn - bnmeani # (batch_size, n_hidden) 
print(f'bndiff.shape: {bndiff.shape}')
bndiff2= bndiff**2 # (batch_size, n_hidden)
print(f'bndiff2.shape: {bndiff2.shape}')
bnvar= 1/(n-1) * bndiff2.sum(dim=0, keepdim=True) # (1, n_hidden)
print(f'bnvar.shape: {bnvar.shape}')
bnvar_inv= (bnvar + 1e-5)**(-0.5) # (1, n_hidden)
print(f'bnvar_inv.shape: {bnvar_inv.shape}')
bnraw= bndiff * bnvar_inv # (batch_size, n_hidden)
print(f'bnraw.shape: {bnraw.shape}')
hpreact= bngain * bnraw + bnbias # (batch_size, n_hidden) and bnbias is broadcasted to (batch_size, n_hidden)
print(f'hpreact.shape: {hpreact.shape}')

#non linear activation layer
h= torch.tanh(hpreact) # (batch_size, n_hidden)
print(f'h.shape: {h.shape}')

#linear layer 2
logits= h @ W2 +b2 # (batch_size, vocab_size) and b2 is broadcasted to (batch_size, vocab_size)
print(f'logits.shape: {logits.shape}')

#cross entropy loss same as F.cross_entropy(logits, Yb)
logit_maxes= logits.max(dim=1, keepdim=True).values # (batch_size, 1)
norm_logits= logits - logit_maxes # (batch_size, vocab_size)
counts=  norm_logits.exp() # (batch_size, vocab_size) 
counts_sum= counts.sum(dim=1, keepdim=True) # (batch_size, 1) 
counts_sum_inv= counts_sum**(-1) 
probs= counts * counts_sum_inv # (batch_size, vocab_size)
logprobs= probs.log()
loss= -logprobs[range(n),Yb].mean()
print(loss)
# PyTorch backward pass
for p in parameters:
    p.grad= None
for t in [logprobs,probs,counts_sum_inv,counts_sum,counts,norm_logits,logit_maxes,logits,h,hpreact,bnraw,
          bnvar_inv,bnvar,bndiff2,bndiff,bnmeani,hprebn,embcat,emb]:
    t.retain_grad()
loss.backward()
print(loss)

#backpropagation by hand
dlogprobs= torch.zeros_like(logprobs)
dlogprobs[range(n),Yb]= -1.0/n
cmp('logprobs',dlogprobs,logprobs)

dprobs= (1.0 / probs) * dlogprobs
cmp('probs',dprobs,probs)

dcounts_sum_inv= (counts * dprobs).sum(dim=1, keepdim=True)
cmp('counts_sum_inv',dcounts_sum_inv,counts_sum_inv)

