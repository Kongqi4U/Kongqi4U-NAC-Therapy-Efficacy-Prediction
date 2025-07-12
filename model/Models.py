import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        
        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out


class COI(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.inc = inc
        self.conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(inc, inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class MHMC(nn.Module):
    def __init__(self, dim, ca_num_heads=4, expand_ratio=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.expand_ratio = expand_ratio

        ca_attention_head_dim = max(32, dim // ca_num_heads)
        ca_attention_expanded_dim = ca_attention_head_dim * ca_num_heads

        self.ca_attention_dim = ca_attention_expanded_dim
        self.ca_attention_head_dim = ca_attention_head_dim

        self.ca_num_heads = self.ca_attention_dim // self.ca_attention_head_dim

        self.temperature = nn.Parameter(torch.ones(self.ca_num_heads, 1, 1))

        self.q = nn.Linear(dim, self.ca_attention_dim, bias=False)
        self.k = nn.Linear(dim, self.ca_attention_dim, bias=False)
        self.v = nn.Linear(dim, self.ca_attention_dim, bias=False)
        self.proj = nn.Linear(self.ca_attention_dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        mlp_hidden_dim = int(dim * expand_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, h, w):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.ca_num_heads, self.ca_attention_head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.ca_num_heads, self.ca_attention_head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.ca_num_heads, self.ca_attention_head_dim).permute(0, 2, 1, 3)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.ca_attention_dim)

        s_out_mlp = self.mlp(x).view(B, h, w, C).permute(0, 3, 1, 2)
        s_out_mlp = F.adaptive_avg_pool2d(s_out_mlp, (1, 1)).view(B, C, 1)
        s_out_final = s_out_mlp.reshape(B, C, N).permute(0, 2, 1)

        x = s_out_final * v.permute(0, 2, 1, 3).reshape(B, N, self.ca_attention_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MAFM(nn.Module):
    def __init__(self, inc, mhmc_ca_num_heads=4, mhmc_expand_ratio=2):
        super().__init__()
        self.outc = inc
        self.attention = MHMC(dim=inc, ca_num_heads=mhmc_ca_num_heads, expand_ratio=mhmc_expand_ratio)
        self.coi = COI(inc)
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1, stride=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )
        self.pre_att = nn.Sequential(
            nn.Conv2d(inc * 2, inc * 2, kernel_size=3, padding=1, groups=inc * 2),
            nn.BatchNorm2d(inc * 2),
            nn.GELU(),
            nn.Conv2d(inc * 2, inc, kernel_size=1),
            nn.BatchNorm2d(inc),
            nn.GELU()
        )

    def forward(self, x, d):
        B, C, H, W = x.shape
        x_cat = torch.cat((x, d), dim=1)
        x_pre = self.pre_att(x_cat)

        x_reshape = x_pre.flatten(2).permute(0, 2, 1)
        attention_out_seq = self.attention(x_reshape, H, W)
        attention_out_img = attention_out_seq.permute(0, 2, 1).reshape(B, C, H, W)

        x_conv = self.coi(attention_out_img)
        x_conv = self.pw(x_conv)

        return x_conv


class AttU_Net_with_MAFM(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, mhmc_ca_num_heads=4, mhmc_expand_ratio=2, extract_features=False):
        super(AttU_Net_with_MAFM, self).__init__()
        self.extract_features = extract_features

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.MAFM5 = MAFM(inc=filters[3], mhmc_ca_num_heads=mhmc_ca_num_heads, mhmc_expand_ratio=mhmc_expand_ratio)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.MAFM4 = MAFM(inc=filters[2], mhmc_ca_num_heads=mhmc_ca_num_heads, mhmc_expand_ratio=mhmc_expand_ratio)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.MAFM3 = MAFM(inc=filters[1], mhmc_ca_num_heads=mhmc_ca_num_heads, mhmc_expand_ratio=mhmc_expand_ratio)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.MAFM2 = MAFM(inc=filters[0], mhmc_ca_num_heads=mhmc_ca_num_heads, mhmc_expand_ratio=mhmc_expand_ratio)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        mafm5_out = self.MAFM5(x4, d5)
        d5 = mafm5_out

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        mafm4_out = self.MAFM4(x3, d4)
        d4 = mafm4_out

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        mafm3_out = self.MAFM3(x2, d3)
        d3 = mafm3_out

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        mafm2_out = self.MAFM2(x1, d2)
        d2 = mafm2_out

        out = self.Conv(d2)

        if self.extract_features:
            features = {
                'mafm2': mafm2_out,
                'mafm3': mafm3_out,
                'mafm4': mafm4_out,
                'mafm5': mafm5_out
            }
            return out, features
        else:
            return out