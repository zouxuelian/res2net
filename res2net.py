class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, stride=1, padding=1, bias=False,groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * self.kernel_size * self.kernel_size, 1,bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        # self.bias = self.bias.repeat(b)
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        # print('weight',weight.shape)
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        # print('weight',weight.shape)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias, stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split

class Bottle2neck(nn.Module):

    def __init__(self, inplanes,stride, scale=4, stype='normal'):

        super(Bottle2neck, self).__init__()

        self.split_out_channels = split_layer(inplanes, scale)
        # print('self.split_out_channels=',self.split_out_channels)#self.split_out_channels= [16, 16, 16, 16]

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(DynamicDWConv(self.split_out_channels[i], kernel_size = 2*i+3, stride=stride, padding=i+1, bias=False,groups=self.split_out_channels[i], reduction=4))
            bns.append(nn.BatchNorm2d(self.split_out_channels[i]))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        spx = torch.split(x, self.split_out_channels, dim=1)
        spk = []
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          spk.append(sp)

        out = torch.cat([s for  s in (spk)], dim=1)
          # out = torch.cat([sp], dim=1)
        print('out',out.shape)

        return out


# if __name__ == '__main__':
#     import torch.onnx
#     import netron
#
#     #####hourglass是自己的网络代码，自己定义的网络结构类名
#     pose = Bottle2neck(64)  # .cuda()
#     dummy_input = torch.randn(1, 64,32, 32)
#     # 输出的文件名称，一般是在当前定义网络路径下
#     onnx_path = "pose.onnx"
#     torch.onnx.export(pose, dummy_input, "pose.onnx")  # netron --host=localhost
#     # 自动跳转到netron的网址下
#     netron.start(onnx_path)

b = torch.randn(2,64,12,12)
a = Bottle2neck(64,1,2)
c = a(b)
print(a)
