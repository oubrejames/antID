import torch.nn as nn 
from ultralytics import YOLO

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv_layer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.conv_layer8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5)
        self.max_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # self.fc1 = nn.Linear(50176 , 256)
        self.fc1 = nn.Linear(57600 , 256) 

        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256 , 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)

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

        out = self.conv_layer7(out)
        out = self.conv_layer8(out)
        out = self.max_pool4(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu1(out)
        out = self.fc4(out)
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

class EN2(nn.Module):
    def __init__(self):
        super(EN2, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 7), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 9),
                                     nn.MaxPool2d(2, stride=2))#,
                                    #  nn.Conv2d(64, 128, 9), nn.PReLU(),
                                    #  nn.MaxPool2d(3, stride=2))

        self.fc = nn.Sequential(nn.Linear(102400 , 256), #518400
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

class EN3(nn.Module):
    def __init__(self):
        super(EN3, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(23104 , 256), #518400
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

class TransferYOLO(nn.Module):
    def __init__(self):
        super(TransferYOLO, self).__init__()
        yolo= YOLO("YOLO_V8/runs/detect/yolov8s_v8_25e6/weights/best.pt")
        yolo = yolo.model.__dict__["_modules"]["model"]
        self.new_yolo = nn.Sequential(*list(yolo.children())[:-3])

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(32, 32, 5), nn.PReLU(),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(32, 64, 5),
                                nn.MaxPool2d(2, stride=2),
                                nn.Conv2d(64, 64, 5), nn.PReLU(),
                                nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(102400 , 256), #518400
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128)
                                )
    def forward(self, x):
        x = x.squeeze(0)
        output = self.new_yolo(x)
        output = self.convnet(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class CNN_EN(nn.Module):
    def __init__(self):
        super(CNN_EN, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
                        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
                        nn.MaxPool2d(kernel_size = 2, stride = 2),
                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
                        nn.MaxPool2d(kernel_size = 2, stride = 2),

                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5),
                        nn.MaxPool2d(kernel_size = 2, stride = 2),

                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7),
                        nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.fc = nn.Sequential(nn.Linear(25088 , 256),
                                nn.ReLU(),
                                nn.Linear(256 , 128))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EN4(nn.Module):
    def __init__(self):
        super(EN4, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 7), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 9),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(#nn.Linear(102400, 256), #for bigger imgs
                                nn.Linear(48256  , 256), #518400 102400
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 128),
                                # nn.PReLU(),
                                # nn.Dropout(p=0.2),
                                # nn.Linear(128, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EN5(nn.Module):
    def __init__(self):
        super(EN5, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 7), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 9),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(48256 , 256), #518400
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class EN6(nn.Module):
    def __init__(self):
        super(EN6, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 7), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 9), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 64, 11),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(39424   , 256), #518400 102400
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(128, 128)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class AntEmbeddingNet(nn.Module):
    def __init__(self):
        super(AntEmbeddingNet, self).__init__()
        
        # Define the architecture
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.fc_block = nn.Sequential(
            # nn.Linear(256 * 28 * 10, 512),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_block(x)
        return x