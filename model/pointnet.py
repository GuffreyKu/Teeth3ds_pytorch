import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              k_size, padding=padding,
                              bias=False)
        
        self.batchNorm = nn.BatchNorm1d(out_channels)
        self.actfunction = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.actfunction(x)
        x = self.batchNorm(x)
        
        return x
    
class MlpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels,  bias=False)
        
        self.batchNorm = nn.BatchNorm1d(out_channels)
        self.actfunction = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.actfunction(x)
        x = self.batchNorm(x)
        
        return x
    

class Tnet(nn.Module):
    def __init__(self, k=64):
        super(Tnet, self).__init__()
        
        self.conv1 = CNNBlock(k, 64, 1)
        self.conv2 = CNNBlock(64, 128, 1)
        self.conv3 = CNNBlock(128, 1024, 1)

        self.fc1 = MlpBlock(1024, 512)
        self.fc2 = MlpBlock(512, 256)

        self.fc3 = nn.Linear(256, k * k)

        self.k = k

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = x.view(-1, self.k, self.k)

        return x
       
class Pointnet(nn.Module):
    def __init__(self, n_points, n_cls):
        super().__init__()
        
        self.conv1 = CNNBlock(in_channels=3, out_channels=64, k_size=1, padding=0)
        
        
        self.tnet2 = Tnet(64)

        self.conv2 = nn.Sequential(
            CNNBlock(in_channels=64,out_channels=128,k_size=1,padding=0),
            CNNBlock(in_channels=128,out_channels=1024,k_size=1,padding=0),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            CNNBlock(in_channels=(1024+64),out_channels=512,k_size=1,padding=0),
            CNNBlock(in_channels=512,out_channels=256,k_size=1,padding=0),
            nn.Dropout(0.3)
        )

        self.conv4 = nn.Conv1d(256, n_cls,1, padding=0)
        self.n_point = n_points
        self.n_cls = n_cls


    def forward(self, x):# B, N, 3(xyz)

        x = x.transpose(2, 1) # B, 3(xyz), N

        x = self.conv1(x) # B, 64, N

        trans_feat = self.tnet2(x) # B, 64, 64
        x = x.transpose(2, 1)# B, N, 64

        x = torch.bmm(x, trans_feat)# B, N, 64
        
        x = x.transpose(2, 1)# B, 64, N
        pointfeat = x
        x = self.conv2(x) # B, 1024, N
        
        x = torch.max(x, 2, keepdim=True)[0] # B, 1024
        x = x.view(-1, 1024, 1).repeat(1, 1, self.n_point) # B, 1024, N

        x = torch.cat([x, pointfeat], 1)# B, 1088(1024+64), N
        x = self.conv3(x) # B, 256, N
        x = self.conv4(x) # B, cls, N
    
        x = x.transpose(2,1).contiguous()# B, N, cls

        return x, trans_feat


class Pointnet_glm(nn.Module):
    def __init__(self, n_points, n_cls):
        super().__init__()
        
        self.conv1 = CNNBlock(in_channels=3, out_channels=64, k_size=1, padding=0)
        
        
        self.tnet2 = Tnet(64)

        self.glm_1 = nn.Sequential(
            CNNBlock(in_channels=64,out_channels=128,k_size=1,padding=0),
        )

        self.glm_2 = nn.Sequential(
            CNNBlock(in_channels=64,out_channels=128,k_size=1,padding=0),
        )

        self.glm_3 = nn.Sequential(
            CNNBlock(in_channels=128+128,out_channels=64,k_size=1,padding=0),
        )

        self.conv2 = nn.Sequential(
            CNNBlock(in_channels=64,out_channels=128,k_size=1,padding=0),
            CNNBlock(in_channels=128,out_channels=1024,k_size=1,padding=0),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            CNNBlock(in_channels=(1024+64+128),out_channels=512,k_size=1,padding=0),
            CNNBlock(in_channels=512,out_channels=256,k_size=1,padding=0),
            nn.Dropout(0.3)
        )

        self.conv4 = nn.Conv1d(256, n_cls,1, padding=0)
        self.n_point = n_points
        self.n_cls = n_cls


    def forward(self, x, adj):# B, N, 3(xyz)

        x = x.transpose(2, 1) # B, 3(xyz), N
        x = self.conv1(x) # B, 64, N

        trans_feat = self.tnet2(x) # B, 64, 64
        x = x.transpose(2, 1)# B, N, 64

        x = torch.bmm(x, trans_feat)# B, N, 64
        x = x.transpose(2, 1)# B, 64, N
        
        sap = torch.bmm(x, adj)# B, 64, N
        glm_feat = self.glm_1(sap)

        x = self.glm_2(x)
        x_glm = x

        x = torch.cat([x, glm_feat], 1)
        x = self.glm_3(x)

        pointfeat = x
        x = self.conv2(x) # B, 1024, N
        
        x = torch.max(x, 2, keepdim=True)[0] # B, 1024
        x = x.view(-1, 1024, 1).repeat(1, 1, self.n_point) # B, 1024, N

        x = torch.cat([x, pointfeat, x_glm], 1)# B, 1216(1024+64+128), N

        x = self.conv3(x) # B, 256, N
        x = self.conv4(x) # B, cls, N
    
        x = x.transpose(2,1).contiguous()# B, N, cls

        return x
    
if __name__ == "__main__":
    device = torch.device("cpu")

    input = torch.randn(16, 100, 3).to(device)
    model = Pointnet_glm(n_points=100, n_cls=5).to(device)

    adj = torch.randn(16, 100, 100).to(device)
    pred = model(input, adj)
    print(pred.size())