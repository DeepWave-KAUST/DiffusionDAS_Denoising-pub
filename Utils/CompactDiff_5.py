import torch
import numpy
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        
        y = self.avg_pool(x).view(b, c)
        #print(y.shape)
        y = self.fc(y).view(b, c, 1)
        return x + (x * y.expand_as(x))
    
class CompactLayer(nn.Module):

    def __init__(self,D,D1):
        super(CompactLayer, self).__init__()    
    
        self.D = D
        self.D1 = D1
        self.con=2
        self.fc1 = nn.Linear(self.D, self.D1) 
        self.ac1 = nn.ReLU()        
        self.fc2 = nn.Linear(self.D1, self.D1)
        self.ac2 = nn.ReLU() 

        self.fc3 = nn.Linear(self.D, self.D1) 
        self.ac3 = nn.ReLU()        
        self.fc4 = nn.Linear(self.D1, self.D1)
        self.ac4 = nn.ReLU() 
        
        self.Att = Attention(self.con)
        
    def forward(self, input, time_emb):
        
        
        x1 = self.ac1(self.fc1(input[:, 0, :]))
        x1 = x1 + time_emb
        x2 = self.ac2(self.fc2(x1))
        #x2 = x2 + x1
        
        x3 = self.ac3(self.fc3(input[:, 1, :]))
        # Adding time embeddings to the respective outputs
        #x3 = x3 + time_emb
        x4 = self.ac4(self.fc4(x3))
        #x4 = x4 + x3

        x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
        x4 = x4.reshape(x4.shape[0], 1, x4.shape[1])
        x = torch.cat((x2, x4), dim=1)

        x = self.Att(x)

        return x
class CompactLayerC(nn.Module):

    def __init__(self,D,D1):
        super(CompactLayerC, self).__init__()    
    
        self.D = D
        self.D1 = D1
        self.con=2
        self.fc1 = nn.Linear(self.D, self.D1) 
        self.ac1 = nn.ReLU()        
        self.fc2 = nn.Linear(self.D1, self.D1)
        self.ac2 = nn.ReLU() 

        self.fc3 = nn.Linear(self.D, self.D1) 
        self.ac3 = nn.ReLU()        
        self.fc4 = nn.Linear(self.D1, self.D1)
        self.ac4 = nn.ReLU() 
        
        self.Att = Attention(self.con)
        
    def forward(self, input, time_emb):
        
        
        x1 = self.ac1(self.fc1(input[:, 0, :]))
        x1 = x1 + time_emb
        x2 = self.ac2(self.fc2(x1))
        #x2 = x2 + x1
        
        x3 = self.ac3(self.fc3(input[:, 1, :]))
        # Adding time embeddings to the respective outputs
        #x3 = x3 + time_emb
        x4 = self.ac4(self.fc4(x3))
        x4 = x4 + x3

        x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
        x4 = x4.reshape(x4.shape[0], 1, x4.shape[1])
        x = torch.cat((x2, x4), dim=1)

        x = self.Att(x)
        
        return x
class DLModel(nn.Module):
    def __init__(self, D, D1):
        super(DLModel, self).__init__()
        self.D = D
        self.D1 = D1
        self.D2 = int(self.D1 / 2)
        self.D3 = int(self.D2 / 2)

        self.E1 = CompactLayerC(self.D, self.D1)
        self.E2 = CompactLayerC(self.D1, self.D2)
        self.E3 = CompactLayerC(self.D2, self.D3)

        self.R1 = CompactLayerC(self.D3, self.D2)
        self.R2 = CompactLayerC(self.D2, self.D1)
        self.R3 = CompactLayerC(self.D1, self.D1)

        self.Fo = nn.Linear(2 * self.D1, self.D)
        self.Foo = nn.Linear(self.D, self.D)


    def forward(self, input1, input2, timesteps):
        # Convert timesteps to tensor if they are not already
        if isinstance(timesteps, int):  # Check if timesteps is an int
            timesteps = torch.tensor([timesteps]).to(input1.device)  # Convert to tensor on the correct device
        else:
            timesteps = timesteps.to(input1.device)  # Ensure timesteps is on the same device as input1
        
        # Calculate time embeddings
        time_emb1 = timestep_embedding(timesteps, self.D1)
        time_emb2 = timestep_embedding(timesteps, self.D2)
        time_emb3 = timestep_embedding(timesteps, self.D3)
        time_emb4 = timestep_embedding(timesteps, self.D2)
        time_emb5 = timestep_embedding(timesteps, self.D1)
        time_emb6 = timestep_embedding(timesteps, self.D1)

        input = torch.cat([input1, input2], dim=1)
        input = input.reshape(input.shape[0], input.shape[1], input.shape[2]*input.shape[3])
        
        E1 = self.E1(input, time_emb1)
        E2 = self.E2(E1, time_emb2)
        E3 = self.E3(E2, time_emb3)

        R1 = self.R1(E3, time_emb4)
        R2 = self.R2(R1, time_emb5)
        R3 = self.R3(R2, time_emb6)

        output = R3.reshape(R3.shape[0], R3.shape[1] * R3.shape[2])
        output = self.Fo(output)
        #output = self.Foo(output)

        output = output.reshape(input1.shape[0], input1.shape[1], input1.shape[2],input1.shape[3])
        return output
    
# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)   
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)  # Ensure to use the correct device

    # Expand timesteps to match the shape for broadcasting
    args = timesteps[:, None].float() * freqs[None]  # Shape: [batch_size, half]

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # Shape: [batch_size, dim]
    
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # Handle the case where dim is odd

    return embedding  # Shape will be [batch_size, dim]
