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

#model definition using classes for layers and activations
class linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight=(torch.randn((fan_in,fan_out),generator=g)/fan_in**0.5).to(device) #kaiming initialisation: weights are scaled by 1/sqrt(fan_in) to maintain the variance of the activations across layers. This helps prevent vanishing/exploding gradients during training. 
        self.bias=torch.randn(fan_out,generator=g).to(device) if bias else None
    def __call__(self,x):
        self.out=x@self.weight
        if self.bias is not None:
            self.out +=self.bias
        return self.out
    def parameters(self):
        return [self.weight]+ ([] if self.bias is None else [self.bias])    

class batchnorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps 
        self.momentum = momentum
        self.training = True
        #parameters trained with backpropagation
        self.gamma=torch.ones(dim).to(device) #scale parameter initialized to 1
        self.beta=torch.zeros(dim).to(device) #shift parameter initialized to 0
        #buffers that are updated with a moving average
        self.running_mean=torch.zeros(dim).to(device)
        self.running_var=torch.ones(dim).to(device)
    def __call__(self,x):
        #calculate the forward pass of batch normalization
        if self.training:
            xmean=x.mean(0,keepdim=True)
            xvar=x.var(0,keepdim=True)
        else:
            xmean=self.running_mean
            xvar=self.running_var
        xhat=(x-xmean)/torch.sqrt(xvar+self.eps) #normalise the input batch to have zero mean and unit variance
        self.out=self.gamma*xhat+self.beta #scale and shift the normalized input using the learnable parameters gamma and beta
        # calculate the running mean and variance for use during evaluation
        if self.training:
            with torch.no_grad():
                self.running_mean=self.momentum*xmean+(1-self.momentum)*self.running_mean
                self.running_var=self.momentum*xvar+(1-self.momentum)*self.running_var
        return self.out
    def parameters(self):
        return [self.gamma,self.beta]   
    
class tanh:
    def __call__(self,x):
        self.out=torch.tanh(x)
        return self.out
    def parameters(self):
        return [] #tanh has no learnable parameters
    


n_embed=10 #size of character embedding vector
vocab_size=27
block_size=3 #context length: how many characters do we look at when predicting the next one?
n_hidden=100 #number of neurons in the hidden layer

g=torch.Generator().manual_seed(2147483647) #for reproducibility
C= torch.randn((vocab_size,n_embed),generator=g).to(device) #character embedding table

#model definition
layers=[
    linear(n_embed*block_size,n_hidden,bias=False),
    batchnorm1d(n_hidden),
    tanh(),
    linear(n_hidden,n_hidden,bias=False),
    batchnorm1d(n_hidden),
    tanh(),
    linear(n_hidden,n_hidden,bias=False),
    batchnorm1d(n_hidden),
    tanh(),
    linear(n_hidden,n_hidden,bias=False),
    batchnorm1d(n_hidden),
    tanh(),
    linear(n_hidden,vocab_size,bias=False),
    batchnorm1d(vocab_size)
]

with torch.no_grad():
    layers[-1].gamma *=0.1 #make the last layer less confident, so that the softmax doesn't produce a one-hot distribution and the loss is not NaN
    for layer in layers[:-1]:
        if isinstance(layer,linear):
            layer.weight *= 1 #scale the weights of linear layers by 5/3 

parameters=[C] + [p for layer in layers for p in layer.parameters()] #collect all parameters from the model into a single list
for p in parameters:
    p.requires_grad=True #set requires_grad to True for all parameters so that we can compute gradients during backpropagation

#training loop
max_steps=200000
batch_size=32
lossi=[]
ud=[]

for i in range(max_steps):

    #construct a mini-batch of data
    ix=torch.randint(0,Xtr.shape[0], (batch_size,), generator=g) #randomly sample batch_size indices from the training set
    Xb, Yb = Xtr[ix], Ytr[ix] #get the corresponding input and target batches

    #forward pass
    emb=C[Xb] #lookup the character embeddings for the input batch. shape: (batch_size, block_size, n_embed)
    x=emb.view(emb.shape[0],-1) #keep the batch dimension and flatten the rest of the dimensions to create a 2D tensor of shape (batch_size, n_embed*block_size)
    for layer in layers:
        x=layer(x)
    loss=F.cross_entropy(x,Yb)

    #backward pass

    for p in parameters:
        p.grad=None #zero the gradients for all parameters before backpropagation
    loss.backward() #compute the gradients of the loss with respect to all parameters   

    #update the parameters using gradient descent
    lr=0.1 if i<150000 else 0.01 #learning rate schedule: start with a higher learning rate and then reduce it after 100k steps
    for p in parameters:
        p.data += -lr*p.grad #update each parameter by moving it in the direction of the negative gradient scaled by the learning rate  
    
    #track the loss and update distance for plotting
    if i%10000==0:
        print(f'{i:7d}/{max_steps}: {loss.item():.4f}') #print the current step and loss every 10k steps
    lossi.append(loss.log10().item()) #append the log of the loss to the list for plotting
    with torch.no_grad():
        ud.append([((lr*p.grad).std()/p.data.std().log10().item()) for p in parameters]) #append the update distance (the ratio of the standard deviation of the parameter updates to the standard deviation of the parameters) to the list for plotting
    
@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  x = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, y)
  print(split, loss.item())

# put layers into eval mode
for layer in layers:
  layer.training = False
split_loss('train')
split_loss('val')   

# words= open('names.txt', 'r').read().splitlines()
# chars=sorted(list(set(''.join(words))))
# stoi={s:i+1 for i,s in enumerate(chars)}
# stoi['.']=0
# itos={i:s for s,i in stoi.items()}

# # sample from the model
# g = torch.Generator().manual_seed(2147483647 + 10)

# for _ in range(20):
    
#     out = []
#     context = [0] * block_size # initialize with all zeros, which corresponds to the '.' token in our vocabulary.
#     # This means that we start with an empty context when generating a new word.
#     while True:
#       # forward pass the neural net
#       emb = C[torch.tensor([context])] # (1,block_size,n_embd)
#       x = emb.view(emb.shape[0], -1) # concatenate the vectors
#       for layer in layers:
#         x = layer(x)
#       logits = x
#       probs = F.softmax(logits, dim=1)
#       # sample from the distribution
#       ix = torch.multinomial(probs, num_samples=1, generator=g).item()
#       # shift the context window and track the samples
#       context = context[1:] + [ix]
#       out.append(ix)
#       # if we sample the special '.' token, break
#       if ix == 0:
#         break
    
#     print(''.join(itos[i] for i in out)) # decode and print the generated word