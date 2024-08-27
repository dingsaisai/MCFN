import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import random
from math import ceil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageEmbedder(nn.Module):
    def __init__(self,dim_inpout, dim, dropout = 0.):
        super().__init__()
        self._fc = nn.Sequential(nn.Linear(dim_inpout, dim), nn.ReLU())
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self._fc(x)
        return self.dropout(x)


class MCFN(nn.Module):
    def __init__(self, in_feats,n_hidden,dropout,omic_sizes,keep_rate,fusion,out_classes):
        super(MCFN, self).__init__()
      
        self.image_embedder= ImageEmbedder(dim_inpout=in_feats,dim=n_hidden, dropout=dropout)
        omic_sizes = omic_sizes
        self.keep_rate=keep_rate
        
        self.fusion = fusion
        
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=n_hidden)]
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.co_attention = MAB(n_hidden, n_hidden, n_hidden, num_heads=8,ln=True)
        
        self.MLP_Mixer = SNN_Mixer(num_blocks=1, node_number=3, node_feature=n_hidden,
                            tokens_mlp_dim=16, channels_mlp_dim=n_hidden//4)
        self.path_rho_1 = nn.Sequential(*[nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)])
        self.path_attention_head = nn.Sequential(nn.LayerNorm(n_hidden),Attn_Net_Gated(L=n_hidden, D=n_hidden//4, dropout=dropout, n_classes=1))
        self.path_rho_2 = nn.Sequential(*[nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)])
        
        self.omic_attention_head = nn.Sequential(nn.LayerNorm(n_hidden),Attn_Net_Gated(L=n_hidden, D=n_hidden//4, dropout=dropout, n_classes=1))
        self.omic_rho = nn.Sequential(*[nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout)])
        
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = GatedBiomodal(n_hidden)
        else:
            self.mm = None


        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, out_classes)
        )

    def forward(self, all_data):
        x_img = all_data.x_img.to(device).float()
        x_cnv=all_data.x_cnv.to(device).float()
        x_mut=all_data.x_mut.to(device).float()
        x_rna=all_data.x_rna.to(device).float()

        x_omic = [x_cnv,x_mut,x_rna]
        
        h_path_bag = self.image_embedder(x_img).unsqueeze(0)

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]  ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)

        h_path_coattn, h_omic_coattn,A = self.co_attention(h_path_bag,h_omic_bag)

        h_path_coattn=ODA(h_path_coattn,A, self.keep_rate)
        h_path_bag = self.path_rho_1(h_path_coattn)
        
        A_path, h_path = self.path_attention_head(h_path_bag.squeeze())
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho_2(h_path).squeeze()

        h_omic_bag =self.MLP_Mixer(h_omic_coattn)
        
        A_omic, h_omic = self.path_attention_head(h_omic_bag.squeeze())
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        if self.fusion == 'bilinear':
            h = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))

        logits = self.classifier(h).unsqueeze(0)

        Y_prob = F.softmax(logits, dim=1)

        return logits,Y_prob




class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False,dropout=0.25):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v1 = nn.Linear(dim_Q, dim_V)
        self.fc_v2 = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln_q = nn.LayerNorm(dim_V)
            self.ln_k = nn.LayerNorm(dim_V)
            
            
        self.fc_q = nn.Sequential(*[nn.Linear(dim_V, dim_V), nn.ReLU(), nn.Dropout(dropout)])
        self.fc_k = nn.Sequential(*[nn.Linear(dim_V, dim_V), nn.ReLU(), nn.Dropout(dropout)])
       
        
    def forward(self, Q, K):
        
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V1, V2 = self.fc_v1(Q), self.fc_v2(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V1_ = torch.cat(V1.split(dim_split, 2), 0)
        V2_ = torch.cat(V2.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        A_T = A.transpose(1, 2)

        O1 = torch.cat((V1_+ A.bmm(V2_)).split(Q.size(0), 0), 2)
        O2 = torch.cat((V2_ + A_T.bmm(V1_)).split(K.size(0), 0), 2)

        O1 = O1 if getattr(self, 'ln_q', None) is None else self.ln_q(O1)
        O1 = self.fc_q(O1)
        
        O2 = O2 if getattr(self, 'ln_k', None) is None else self.ln_k(O2)
        O2 = self.fc_k(O2)
        
        return O1,O2,A_T
def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

def ODA(h_path, A,keep_rate):
    """
    Args:
        h_path: the feature for instance
        A: attention score for h_path
        keep_rate: The keep rate of attentive instance
    """
    # attention matrix normalization
    cls_attn = A.mean(dim=0)
    cls_attn = cls_attn.mean(dim=0)
    cls_attn = cls_attn.unsqueeze(0)

    # instance decoupling
    N = cls_attn.shape
    left_tokens = math.ceil(keep_rate * (N[1]))

    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)


    #inattentive instances fusion
    compl = complement_idx(idx, N[1])  # [N-1-left_tokens]
    x_inatten = torch.gather(h_path, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, 256))  # [B, N-1-left_tokens, C]
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
    x_inatten = torch.sum(x_inatten * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)


    # attentive matching
    index = idx.unsqueeze(-1).expand(-1, -1, 256)  # [left_tokens, C]
    x_atten = torch.gather(h_path, dim=1, index=index)

    vectors = x_atten.squeeze(0)

    
    norms = vectors.norm(dim=1, keepdim=True)

    normalized_vectors = vectors / norms
    # 计算余弦相似度矩阵
    cosine_similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.t())
    # 将对角线上的值设为0
    cosine_similarity_matrix.fill_diagonal_(0)
    # 找到每行的最大值及其索引
    max_list = torch.argmax(cosine_similarity_matrix, dim=1)

    if len(max_list) > 0:
        add_vectors = vectors[max_list, :]
        new_vectors = 0.5*vectors + 0.5*add_vectors
        x_new = new_vectors.unsqueeze(0).cuda()
        x_path = torch.cat([x_inatten, x_atten,x_new], dim=1)
    else:
        x_path = torch.cat([x_inatten, x_atten], dim=1)

    return x_path

class GatedBiomodal(nn.Module):
    def __init__(self, input_size):
        super(GatedBiomodal,self).__init__()
        self.input_size = input_size
        self.weight1 = Parameter(torch.FloatTensor(input_size, input_size))
        self.weight2 = Parameter(torch.FloatTensor(input_size, input_size))
        self.weight3 = Parameter(torch.FloatTensor(2* input_size, 2* input_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0/ math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)
    
    def forward(self, x1, x2):
        concat = torch.cat([x1,x2],0)
        concat = torch.matmul(concat, self.weight3)
        z = torch.sigmoid(concat)
        x1_tanh = torch.tanh(torch.matmul(x1, self.weight1))
        x2_tanh = torch.tanh(torch.matmul(x2, self.weight2))
        out = z[:self.input_size] *x1_tanh + (1-z[self.input_size:]) *x2_tanh
        return out

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""


    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))
            
class Attn_Net_Gated(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class SNNBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim,dropout=0.25):
        super(SNNBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),  # 
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False),
            nn.Linear(mlp_dim, hidden_dim),            # 
            nn.AlphaDropout(p=dropout, inplace=False)
        )

    def forward(self, x):
        y = self.mlp(x)

        return y



class SNN_MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(SNN_MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = SNNBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = SNNBlock(hidden_dim, channels_mlp_dim)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x



class SNN_Mixer(nn.Module):
    '''tokens_mlp_dim对应的是结点方向的全连接维度的隐藏维度,
    channels_mlp_dim对应的是结点特征维度上进行全连接层的隐藏维度
    num_blocks表示 mlp_block的个数
    '''

    def __init__(self,  num_blocks, node_number, node_feature, tokens_mlp_dim, channels_mlp_dim):
        super(SNN_Mixer, self).__init__()

        num_tokens = node_number

        self.mlp = nn.Sequential(
            *[SNN_MixerBlock(num_tokens, node_feature, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(node_feature)


    def forward(self, x):
        x = self.mlp(x)
        x = self.ln(x)

        return x




