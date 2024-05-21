import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
import torch.utils.data
import torchvision

batch_size = 128
device = "cuda"
num_of_epochs = 10
learning_rate=1e-4
embed_dim = 768
mlp_hidden_size = 4 * embed_dim
num_of_layers = 12
num_of_classes = 10

torch.autograd.set_detect_anomaly(True)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


@torch.no_grad()
def estimate_loss(trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader) -> dict:
    '''
    Esitmating Train and Test split losses.
    '''
    out = {}
    train = trainloader
    test = testloader
    model.eval()
    for split in ['train', 'eval']:
        num_of_items = 0
        running_sum = 0
        data = train if split=="train" else test
        
        for image, cls in data:
            B, _, _, _ = image.shape
            if (B != batch_size):
                break
            image = image.to(device)
            cls = cls.to(device)
            predict, loss = m(image, cls)
            running_sum += loss.item()
            num_of_items +=1
            
        
        if (num_of_items == 0):
            out[split] = 0
        else:
            out[split] = running_sum/num_of_items
    
    model.train()
    return out

class Embedding(nn.Module):
    '''
    Embedding Module for Vision Transformer.
    '''
    
    def __init__(self, image_size: tuple, embed_dim: int=embed_dim, patch_size: int=7, batch_size: int=batch_size, dropout_p: float=0.1):
        super().__init__()
        num_of_patches = int((image_size[0] * image_size[1])/ patch_size**2)
        self.patches_layer = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        
        self.class_token = nn.Parameter(
            torch.ones(batch_size, 1, embed_dim), requires_grad= True
        )
        
        self.position_tokens = nn.Parameter(
            torch.ones(batch_size, num_of_patches+1, embed_dim), requires_grad=True
        )
        
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        patches = self.patches_layer(x)
        patches = nn.Flatten(start_dim=2, end_dim=3)(patches).permute(0, 2, 1)
        patches = torch.concat((self.class_token, patches), dim=1)
        patches += self.position_tokens
        return self.dropout(patches)
    

class Head(nn.Module):
    
    def __init__(self, head_size: int, embed_dim: int=embed_dim, dropout_f: float=0.2):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout_f)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        qk = torch.matmul(q, k.transpose(-2, -1)) * (C ** -0.5)
        wei = F.softmax(qk, dim=-1)
        wei = self.dropout(wei)

        out = torch.matmul(wei, v)
        return out
    
class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, num_heads: int, embed_dim: int=embed_dim, dropout_f: float=0.2):
        super().__init__()
        head_size = embed_dim//num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size* num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout_f)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim= -1)
        out = self.dropout(self.proj(out))
        return out

class MultiLayeredPerceptron(nn.Module):
    
    def __init__(self, mlp_hidden_size: int, droput_p: float=0.1, embed_dim: int=embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(droput_p),
            nn.Linear(mlp_hidden_size, embed_dim),
            nn.Dropout(droput_p),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    

class Block(nn.Module):
    
    def __init__(self, embed_dim: int=embed_dim, mlp_hidden_size: int=mlp_hidden_size, dropout_p: float=0.1): #3072 = 4 * emebed_dim
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(12)
        self.mlp = MultiLayeredPerceptron(mlp_hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln1(x)
        out = out + self.msa(out)
        out = self.ln2(out)
        out = out + self.mlp(x)
        return out
    
class VisionTransformer(nn.Module):
    
    def __init__(self, num_of_layers: int=num_of_layers, num_class: int=num_of_classes, embed_dim: int=embed_dim):
        super().__init__()
        self.embedding_layer = Embedding((28, 28))
        self.transformer_encoder = nn.Sequential(*[Block() for _ in range(num_of_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_class),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        embeds = self.embedding_layer(x)
        predict = self.classifier(self.transformer_encoder(embeds)[:, 0])
        
        if target == None:
            loss = None
        else:
            loss = F.cross_entropy(predict, target)
        
        return predict, loss    
            


model = VisionTransformer()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(num_of_epochs):
    for images, cls in trainloader:
        B, _, _, _ = images.shape
        if (B != batch_size):
            break
        images = images.to(device)
        cls = cls.to(device)
        predict, loss = m(images, cls)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        
    losses = estimate_loss(trainloader, testloader)           
    print(f"step:{iter} : train loss: {losses['train']:.4f} : val loss: {losses['eval']:.4f}")