import torch
import torch.nn as nn

"Architecture based on https://arxiv.org/pdf/1711.10684.pdf"


class BatchNormRelu(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.batch_norm(inputs)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.r = ResidualBlock(input_channels + output_channels, output_channels)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        
        """ Conv layer"""
        self.batch_norm1 = BatchNormRelu(input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1, stride=(stride, stride))
        self.batch_norm2 = BatchNormRelu(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1, stride=(1, 1))

        """ shortcut connection"""
        self.s = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), padding=0, stride=(stride, stride))

    def forward(self, inputs):
        x = self.batch_norm1(inputs)
        x = self.conv1(x)
        x = self.batch_norm2(x)
        x = self.conv2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class ResidualUnet(nn.Module):
    def __init__(self, n_channels=3,n_output_channels=3, need_feature_maps=False):
        super().__init__()
        self.need_feature_maps = need_feature_maps
        
        """ encoder 1 """
        self.conv11 = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), padding=1)
        self.batch_relu1 = BatchNormRelu(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.conv13 = nn.Conv2d(n_channels, 64, kernel_size=(1, 1), padding=0)  # one by one conv

        """ encoder 2 & 3"""
        self.r2 = ResidualBlock(64, 128, stride=2)
        self.r3 = ResidualBlock(128, 256, stride=2)

        """ Bridge """
        self.r4 = ResidualBlock(256, 512, stride=2)

        """ Decoder """
        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, n_output_channels, kernel_size=(1, 1), padding=0)
        self.sigmoid = nn.Sigmoid()  # do not know whether we are going to need it
        
    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, ( nn.Conv2d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv2d,  nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, inputs):
        """ encoder 1"""
        x = self.conv11(inputs)
        x = self.batch_relu1(x)
        x = self.conv12(x)
        s = self.conv13(inputs)  # residual shortcut
        skip1 = x + s  # skip works as a skip connection for a decoder and next encoder block

        """ encoder 2 & 3"""
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ Output """
        output = self.output(d3)
        if self.need_feature_maps:
            return output, d3
        return output
        
#         output = self.sigmoid(output)

        return output


class DewarpingUNet(nn.Module):
    def __init__(self, n_channels, n_output_channels):
        super(DewarpingUNet, self).__init__()
        # U-net1
        self.UNet1 = ResidualUnet(n_channels, n_output_channels, need_feature_maps=True)
        self.UNet2 = ResidualUnet(64 + n_output_channels, n_output_channels, need_feature_maps=False)

    def forward(self, x):
        y1,feature_maps = self.UNet1(x)
        x = torch.cat((feature_maps, y1), dim=1)
        # print("x:",x.shape)
        # print("y1:",y1.shape)
        y2 = self.UNet2(x)
        # print("y2:",y2.shape)
        return y1, y2
    
    
if __name__ == "__main__":
    inputs = torch.randn((4, 3, 256, 256))
    model = ResidualUnet()
    print(model(inputs))