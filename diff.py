import torch
import torch.nn as nn


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_step=10,device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.num_step=num_step

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        '''
        Generate noise beta of each time step
        return: shape (noise_steps, )
        '''
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        '''
        Sample time steps for each image
        input:
            n: batch_size, int
        return:
            t: time_step, shape (n, ), values in [1, noise_step]
        '''
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def noise_images(self, x, t):
        '''
        Add noise process: x_0 -> x_{t}
        input:
            x: input_images, shape (batch_size, 1, img_size, img_size)
            t: time_step, int
        return:
            noise_images: shape (batch_size, 1, img_size, img_size)
            noise: shape (batch_size, 1, img_size, img_size)
            noise_images = sqrt(alpha_hat[t]) * x + sqrt(1 - alpha_hat[t]) * noise
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        Ɛ =torch.randn_like(torch.tensor(x,dtype=torch.float32))
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample(self, model, n, data,num_cls,modal=0):
        '''
        Denoise process: x_{t} -> x_{t-1} -> ... -> x_0
        input:
            model: nn.Module
            n: batch_size, int
            labels: shape (n, ), values in [0, 9]
            cfg_scale: float, 0.0 ~ 1.0, 0.0: unconditioned diffusion, 1.0: conditioned diffusion
        return:
            x_0: images in t0, shape (n, 1, img_size, img_size), values in [0, 255]
            sampled_images (x_{t-1}) = 1 / sqrt(alpha[t]) * (noisy_images (x_t) - (1 - alpha[t]) / sqrt(1 - alpha_hat[t]) * predicted_noise) + sqrt(beta[t]) * noise
        '''


        model.eval()
        with torch.no_grad():
            x_list = []
            for i in range(self.num_step):
                x = torch.randn((n, num_cls)).to(self.device)  # 随机噪声
                for i in range(1, self.noise_steps):
                    t = (torch.ones(n) * i).long().to(self.device)

                    predicted_noise = model(x, t, data)  # 把每个时间步长的t输进去，得到预测
                    # interpolate with unconditioned diffusion

                    alpha = self.alpha[t][:, None]
                    alpha_hat = self.alpha_hat[t][:, None]
                    beta = self.beta[t][:, None]  # 几个参数
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (
                                x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                        beta) * noise  # 对x去噪
                    x_list.append(x.unsqueeze(0))
        model.train()
        if modal==0:
            return torch.mean(torch.cat(x_list, 0), 0)
        else:
            return torch.cat(x_list, 0)





class C_net(nn.Module):
    def __init__(self, num_classes,feature_dim,num_step,act=nn.LeakyReLU(0.1),time_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.num_class=num_classes
        self.num_step=num_step

        self.map=nn.Linear(feature_dim,self.num_class)
        self.map1=nn.Linear(num_step,self.num_class)
        self.fusion=nn.Sequential(nn.Linear(self.num_class*3,self.num_class))
        self.label_emb=nn.Sequential(nn.Linear(self.num_class,feature_dim),nn.Linear(feature_dim,self.num_class))
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** ((torch.arange(0, channels, 2, device=self.device).float()) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, label_t, t, data_iter):

        t = t.unsqueeze(-1).type(torch.float)
        t = self.map1(self.pos_encoding(t, self.num_step))
        data_iter=self.map(data_iter)

        label_t=self.label_emb(label_t)


        return self.fusion(torch.cat([data_iter,label_t,t],1))+data_iter+label_t+t
import random

import torch
import math
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.autograd as ag
class GELU(nn.Module):#zengen
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        #return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GroupAttention(nn.Module):
    def __init__(self, dim, num_heads=4, N_Pi=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1,
                 sr_ratio=1.0):
        """
        ws 1 for stand attention
        """
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        dim = dim // N_Pi  # 光谱token数
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * N_Pi, dim * N_Pi)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    # @auto_fp16()
    def forward(self, x, N_Pi, D_Pi):
        B, N, C = x.shape
        x = x.view(B, N, N_Pi, D_Pi)
        qkv = self.qkv(x).reshape(B, N, N_Pi, 3, self.num_heads,
                                  D_Pi // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3)
        x = attn.reshape(B, N, N_Pi, D_Pi)
        # if pad_r > 0 or pad_b > 0:
        #     x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        # dim = dim // 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio

        # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # x1 = x.reshape(B,-1,16)
        # B, N, C = x1.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
        # x_ = self.sr(x)
        x_ = self.norm(x)
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.drop_path = nn.Identity()  # zhan wei
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class GroupBlock(Block):
    def __init__(self, depth, dim, num_heads, N_Pi, patch_size, local_kiner=3, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        # del self.attn1
        # del self.attn2
        # if ws == 1:
        #     self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        # else:
        #     self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)
        self.depth = depth
        self.attn1 = GroupAttention(dim, num_heads, N_Pi, qkv_bias, qk_scale, attn_drop, drop)
        self.attn2 = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        self.pool = nn.AvgPool2d(local_kiner, 1, (local_kiner - 1) // 2)

        self.patch_size = patch_size
        self.dim = dim



    def forward(self, x, N_Pi, D_Pi, H, W,  GPU=0):
        for i in range(self.depth):
            if i < 0:
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))

            else:
                x = x + self.drop_path(self.attn1(self.norm1(x), N_Pi, D_Pi))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                x = x + self.drop_path(self.attn2(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
class FE(nn.Module):
    def __init__(self, image_size, near_band, num_patches, patch_size, num_classes, dim,dim1, pixel_dim, depth, heads,
                 mlp_dim, pool='cls', channels=1, dim_head=16, dropout=0., emb_dropout=0., mode='ViT', GPU=1,
                 local_kiner=3):
        super().__init__()

        patch_dim = image_size

        self.GPU = GPU
        self.patch_size = patch_size
        self.dim = dim

        self.num_classes = num_classes

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding_p = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dis_cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = GroupBlock(depth=depth, dim=dim, num_heads=heads, N_Pi=pixel_dim, patch_size=self.patch_size,
                                      local_kiner=local_kiner)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim1)
        )

    def forward(self, x):
        x=x.permute(0,2,3,1).reshape(x.shape[0],-1,x.shape[1])

        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]

        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]

        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding_p[:, :]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # x = rearrange(x, 'b n d -> (b n) d')
        # x = x.reshape(-1,16,4)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        N_Pi = 4
        D_Pi = 16
        H = W = 14
        casual= self.transformer(x, N_Pi, D_Pi, H, W,  self.GPU)
        # classification: using cls_token output
        x = self.to_latent(casual[:, 0])
        x=self.mlp_head(x)

        # attention module

        return x
class SS_FE(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim1,hidden_dim2,act,args):
        super().__init__()

        self.output_dim=output_dim

        self.spa_fe=FE_( input_dim=input_dim,output_dim=output_dim,hidden_dim=hidden_dim1,act=act)
        self.spe_fe = FE_(input_dim=args.patches**2, output_dim=output_dim, hidden_dim=hidden_dim2, act=act)
        self.fusin_layer=nn.Sequential(
            nn.Linear(self.output_dim*2,self.output_dim),

                                       act,

                                       nn.Linear(self.output_dim, self.output_dim),
                                       act,

                                       )



    def forward(self, x):
        patch=x.shape[2]
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


        x1=x.reshape(-1,x.shape[-1])
        x2=x.reshape(-1,patch**2,x.shape[-1]).permute(0,2,1).reshape(-1,patch**2)

        x1=self.spa_fe(x1).reshape(-1,patch**2,self.output_dim )
        x2 = self.spe_fe(x2).reshape(-1,x.shape[-1] , self.output_dim)




        x=self.fusin_layer(torch.cat([torch.mean(x1,1),torch.mean(x2,1)],1))


        return x
class FE_(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim,act):
        super().__init__()

        self.input_dim=input_dim
        self.output_dim = output_dim
        self.hiddent_dim=hidden_dim
        self.act=act
        self.layer=nn.Sequential()
        if len(self.hiddent_dim)==0:



            self.layer.append(nn.Linear(self.input_dim, output_dim))
            self.layer.append(self.act)
        else:
            for i in range(len(hidden_dim)):
                if i==0:



                    self.layer.append(nn.Linear(self.input_dim, self.hiddent_dim[0]))

                    self.layer.append(self.act)
                if i!=0 and i!=len(hidden_dim)-1:



                    self.layer.append(nn.Linear(self.hiddent_dim[i-1], self.hiddent_dim[i]))

                    self.layer.append(self.act)
                if i==len(hidden_dim)-1:



                    self.layer.append(nn.Linear(self.hiddent_dim[i-1], self.output_dim))

                    self.layer.append(self.act)




    def forward(self, x):

        x=x.reshape(-1,self.input_dim)

        x=self.layer(x)






        return x
