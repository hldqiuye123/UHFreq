import torch.nn as nn
import torch
from complexLayers import ComplexLinear,ComplexReLU,ComplexConv1d,ComplexConvTranspose1d,ComplexConv2d,ComplexBatchNorm1d
import matplotlib.pyplot as plt
import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 and 'ComplexLinear' not in classname:
        nn.init.orthogonal(m.weight.data)


def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_layer1_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_layer1(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.to(args.device)
    return net

def set_skip_module(args):
    """
    Create a frequency-representation module
    """
    net = None

    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_skiplayer32(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)
    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.to(args.device)

    return net


def set_rdn_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_dense(scale_factor=args.fr_upsampling, num_features=args.fr_n_filters,
                                            num_blocks=9, num_layers=4,
                                            growth_rate=args.fr_n_filters,signal_dim=args.signal_dim,inner_dim=args.fr_inner_dim)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.to(args.device)
    return net



class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        self.lff = nn.Conv1d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)
    # RDN
    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class FrequencyRepresentationModule_dense(nn.Module):
    def __init__(self, scale_factor, num_features, growth_rate, num_blocks, num_layers,signal_dim,inner_dim):
        super().__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.fr_size = inner_dim * scale_factor
        self.n_filters = num_features
        self.in_layer = ComplexLinear(signal_dim, inner_dim * num_features)
        # self.in_layer=ComplexConv1d(1, inner_dim * num_features,kernel_size=(1,64), padding=0, bias=False)
        self.in_layer2 = ComplexConv1d(num_features, num_features, kernel_size=3, padding=3 // 2, bias=False)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * (self.D), self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.ConvTranspose1d(self.G0, 1, 18, stride=16,
                                            padding=1, output_padding=0, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
        # inp=inp.view(bsz,1,1,-1)
        sfe1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        sfe2=self.in_layer2(sfe1)
        x=sfe2.abs()
        # x = sfe2.abs()
        # x=self.mid(x)
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(x[0,i])
        #     plt.show()
        #     plt.pause(1)

        # Fbar=self.ca(sfe1)*sfe1
        # x= sfe1 + self.sa(Fbar)*Fbar

        local_features = []

        for i in range(self.D):
            x1 = self.rdbs[i](x)
            # plt.figure()
            # for i in range(32):
            #     plt.ion()
            #     plt.plot(x[0, i])
            #     plt.show()
            #     plt.pause(1)
            local_features.append(x1)

        # plt.figure()
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(local_features[-1][0, i])
        #     plt.show()
        #     plt.pause(1)
        fx = self.gff(torch.cat(local_features, 1))   # global residual learning
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(fx[0,i])
        #     plt.show()
        #     plt.pause(1)
        # fx+= x
        x=self.output(fx).view(bsz,-1)
        # comd=[]
        # comd.append(local_features[-1])
        # comd.append(sfe1)
        # fx = self.gff2(torch.cat(comd, 1))
        # plt.figure()
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(x[0, i])
        #     plt.show()
        #     plt.pause(1)
        # x=self.output(local_features[-1]).view(bsz,-1)
        # x1=x[:,:,0:2]
        # x2=x[:,:,-1]
        # plt.figure()
        # plt.plot(x[0,:])
        return x

class FrequencyRepresentationModule(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = ComplexLinear(signal_dim, inner_dim * n_filters)

        mod = []
        for n in range(48):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
            self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        # inp = inp.view(bsz, -1)
        inp = inp[:, 0, :].type(torch.complex64) + 1j * inp[:, 1, :].type(torch.complex64)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x=x.abs()
        # plt.figure()
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(x[0,i])
        #     plt.show()
        #     plt.pause(1)
        x = self.mod(x)
        # plt.plot(x[0,0,:].cpu().detach().numpy())
        #         # plt.close()
        x = self.out_layer(x).view(bsz, -1)
        # plt.figure()
        # plt.plot(x[0, :].cpu().detach().numpy())
        # plt.show()
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelAttention, self).__init__()
        # Applies global average pooling over the temporal dimension to aggregate contextual information
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Applies global max pooling over the temporal dimension to highlight prominent features
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # First shared fully connected layer implemented as 1×1 convolution to reduce channel dimensionality
        self.fc1 = nn.Conv1d(num_channels, num_channels // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        # Second shared fully connected layer to restore the original channel dimensionality
        self.fc2 = nn.Conv1d(num_channels // 8, num_channels, 1, bias=False)

    def forward(self, x):
        # Channel attention map from average-pooled features
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # Channel attention map from max-pooled features
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # Aggregates both descriptors and applies sigmoid activation to obtain final attention weights
        out = avg_out + max_out
        # Recalibrates the input features by channel-wise multiplication with the attention weights
        return x * torch.sigmoid(out)



class ChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(num_channels, num_channels // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(num_channels // 8, num_channels, 1, bias=False)

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * torch.sigmoid(out)

class PixelAttention(nn.Module):
    def __init__(self):
        super(PixelAttention, self).__init__()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = avg_out + max_out
        return x * torch.sigmoid(out)
class ComplexChannelAttention(nn.Module):
    def __init__(self, num_channels):
        super(ComplexChannelAttention, self).__init__()
        # 用于实部和虚部的通道注意力
        self.real_channel_attention = ChannelAttention(num_channels)
        self.imag_channel_attention = ChannelAttention(num_channels)

    def forward(self, x):
        # 分离实部和虚部
        real_part, imag_part = x.real, x.imag

        # 分别对实部和虚部应用通道注意力
        real_attended = self.real_channel_attention(real_part)
        imag_attended = self.imag_channel_attention(imag_part)

        # 重新组合为复数
        attended_output = torch.complex(real_attended, imag_attended)
        return attended_output
class ComplexPixelAttention(nn.Module):
    def __init__(self):
        super(ComplexPixelAttention, self).__init__()
        # 实部和虚部的像素注意力
        self.real_pixel_attention = PixelAttention()
        self.imag_pixel_attention = PixelAttention()

    def forward(self, x):
        # 分离实部和虚部
        real_part, imag_part = x.real, x.imag

        # 分别对实部和虚部应用像素注意力
        real_attended = self.real_pixel_attention(real_part)
        imag_attended = self.imag_pixel_attention(imag_part)

        # 重新组合为复数
        attended_output = torch.complex(real_attended, imag_attended)
        return attended_output

class FrequencyRepresentationModule_skiplayer32(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters


        self.inner=inner_dim
        self.n_layers=n_layers
        self.channel_attention = ComplexChannelAttention(n_filters*2)
        self.pixel_attention = ComplexPixelAttention()
        self.in_layer = ComplexLinear(signal_dim, inner_dim * int(n_filters / 8))

        self.in_layer2 = ComplexConv2d(1,  int(n_filters), kernel_size=(1, 3), padding=(0, 3 // 2),
                                       bias=False)
        # Encoder layers
        self.encoders = nn.ModuleList()
        for i in range(self.n_layers):
            encoder_layer = nn.Sequential(
                ComplexConv1d(n_filters , n_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2, stride=2, bias=False),
                ComplexBatchNorm1d(n_filters * 2),
                ComplexReLU()
            )
            self.encoders.append(encoder_layer)
            n_filters *= 2  # Double the number of filters after each layer

        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in reversed(range(self.n_layers-1)):
#             if i!=self.n_layers-1:
            decoder_layer = nn.Sequential(
                ComplexConvTranspose1d(n_filters, n_filters//2, kernel_size=kernel_size, padding=kernel_size // 2, stride=2, output_padding=1, bias=False),
                ComplexBatchNorm1d(n_filters // 2),
                ComplexReLU()
            )
#             else:
#                 decoder_layer = nn.Sequential(
#                     ComplexConv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=False),
#                     ComplexBatchNorm1d(n_filters),
#                     ComplexReLU()
#                 )
            self.decoders.append(decoder_layer)
            n_filters //= 2  # Halve the number of filters after each layer

        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[ComplexReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer1 = ComplexConvTranspose1d(n_filters, n_filters//2, 18, stride=4,padding=7, output_padding=0, bias=False)
        self.out_layer2 = ComplexConvTranspose1d(n_filters//2, 1, 9,stride=1,padding=4, output_padding=0, bias=False)
        self.relu_amp = nn.ReLU()
        self.tanh_phase = nn.Tanh()
    def forward(self, x):
        bsz = x.size(0)
        grid = np.linspace(-0.5, 0.5, 256, endpoint=False)
        inp = x[:, 0, :].type(torch.complex64) + 1j * x[:, 1, :].type(torch.complex64)

        x = self.in_layer(inp)

        x = x.view(bsz, 1, int(self.n_filters / 8), -1)

        x = self.in_layer2(x)

        x = x.view(bsz, self.n_filters, -1)


        encoder_outputs = []
        for encoder in self.encoders:
            x = encoder(x)

            encoder_outputs.append(x)

        for i, decoder in enumerate(self.decoders):
            x = decoder(x)

            x += encoder_outputs[-(i + 2)]

        x = self.channel_attention(x)

        x = self.pixel_attention(x)

        x = self.out_layer1(x)

        x = self.out_layer2(x).view(bsz, -1)

        return x


class FrequencyRepresentationModule_layer1(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling

        self.n_filters = n_filters


        self.inner=inner_dim
        self.n_layers=n_layers

        self.in_layer = ComplexLinear(signal_dim, inner_dim * n_filters)

        # self.in_layer2 = ComplexConv2d(1,  int(n_filters/4), kernel_size=(1, 3), padding=(0, 3 // 2),
        #                                bias=False)
        mod=[]
        for i in range(self.n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(self.n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)

        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 18, stride=16,
                                            padding=1, output_padding=0, bias=False)
        # self.out_layer = nn.ConvTranspose1d(n_filters, 1, 9, stride=8,
        #                                     padding=1, output_padding=1, bias=False)
        # self.out_layer2=nn.Conv1d(n_filters, 1, kernel_size=3, padding=3 // 2, bias=False)




    def forward(self, x):
        bsz = x.size(0)
        grid = np.linspace(-0.5, 0.5, 256, endpoint=False)
        inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)


        x=self.in_layer(inp).view(bsz, self.n_filters, -1)

        #plt.figure()
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        #for i in range(0,4):

            #plt.ion()
            #plt.plot(x[0,0,i].abs()/torch.max(x[0,0].abs()))
            #plt.show()
            #plt.pause(1)
        # x=self.in_layer2(x).view(bsz,self.n_filters,-1)
        # plt.figure()
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.axis('off')
        # for i in range(0,32):
        #
        #     plt.ion()
        #     plt.plot(x[0,i].abs()/torch.max(x[0].abs()))
        #     plt.show()
        #     plt.pause(1)
        x=x.abs()
        # plt.figure()
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.axis('off')
        # for i in range(0,16):
        #
        #     plt.ion()
        #     plt.plot(grid,x[0,i].abs())
        #     plt.show()
        #     plt.pause(1)



        # plt.figure()
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.axis('off')
        # for i in range(0,32):
        #
        #     plt.ion()
        #     plt.plot(grid,x[0,i])
        #     plt.show()
        #     plt.pause(1)

        for i in range(self.n_layers):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)

        # plt.figure()
        # # plt.xticks([])
        # # plt.yticks([])
        # # plt.axis('off')
        # for i in range(0, 16):
        #     plt.ion()
        #     plt.plot(grid,x[0, i])
        #     plt.show()
        #     plt.pause(1)


        x = self.out_layer(x).view(bsz, -1)

        # plt.figure()
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        # grid = np.linspace(-0.5, 0.5, 4096, endpoint=False)
        # plt.plot(grid,x[0],'b')
        # plt.show()
        return x

# class FrequencyRepresentationModule_layer1(nn.Module):
#     # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
#     def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
#                  kernel_size=3, upsampling=8, kernel_out=25):
#         super().__init__()
#         self.fr_size = inner_dim * upsampling
#         self.n_filters = n_filters
#         self.in_layer = ComplexLinear(signal_dim, inner_dim * n_filters)
#
#         mod=[]
#         for i in range(24):
#             tmp = []
#             tmp += [
#                 nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
#                           padding_mode='circular'),
#                 nn.BatchNorm1d(n_filters),
#                 nn.ReLU(),
#                 nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
#                           padding_mode='circular'),
#                 nn.BatchNorm1d(n_filters),
#             ]
#             mod+= [nn.Sequential(*tmp)]
#         self.mod=nn.Sequential(*mod)
#         activate_layer = []
#         for i in range(24):
#             activate_layer+=[nn.ReLU()]
#         self.activate_layer=nn.Sequential(*activate_layer)
#
#         self.out_layer = nn.ConvTranspose1d(n_filters, 1, 9, stride=8,
#                                             padding=1, output_padding=1, bias=False)
#
#
#     def forward(self, x):
#         bsz = x.size(0)
#
#         inp = x[:,0,:].type(torch.complex64)+1j*x[:,1,:].type(torch.complex64)
#         x = self.in_layer(inp).view(bsz, self.n_filters, -1)
#         # x=self.in_layer2(x)
#
#         x=x.abs()
#         for i in range(24):
#             res_x = self.mod[i](x)
#             x = res_x + x
#             x = self.activate_layer[i](x)
#
#
#
#         x = self.out_layer(x).view(bsz, -1)
#         return x
