import torch.nn as nn 

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_layer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(236672 , 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
        
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.max_pool3(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 9), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 9), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(484416 , 256), #518400
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
    
class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 64, 7, stride = 2),
                                     nn.MaxPool2d(3, stride=2),
                                     nn.LocalResponseNorm(2),
                                     nn.Conv2d(64, 64, 1), 
                                     nn.Conv2d(64, 192, 1), 
                                     nn.LocalResponseNorm(2),
                                     nn.MaxPool2d(3, stride=2),
                                     nn.Conv2d(192, 192, 1),
                                     nn.Conv2d(192, 384, 3),
                                     nn.MaxPool2d(3, stride=2),
                                     nn.Conv2d(384, 384, 1),
                                     nn.Conv2d(384, 256, 3),
                                     nn.Conv2d(256, 256, 1),
                                     nn.Conv2d(256, 256, 3),
                                     nn.Conv2d(256, 256, 1),
                                     nn.Conv2d(256, 256, 3),
                                     nn.MaxPool2d(3, stride=2))

        self.fc = nn.Sequential(nn.Linear(12544 , 128*32), #518400
                                nn.PReLU(),
                                nn.Linear(128*32, 128*32),
                                nn.PReLU(),
                                nn.Linear(128*32, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)