"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/21
    Time:上午9:09
    To change this template use File | Settings | File Templates
"""
"""模型结构相关代码"""
import config
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device

"""
将输入的离散词索引转换为连续的向量表示
例如，将词汇表中的第5个词映射为一个512维的向量
"""


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # 初始化方法，传入模型的维度（d_model）和词汇表的大小（vocab）
        super(Embeddings, self).__init__()
        # Embedding层，将词汇表的大小映射为d_model维的向量
        ## 这一层的参数量是: vocab_size * d_model
        self.lut = nn.Embedding(vocab, d_model)
        # 存储模型的维度 d_model
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        # 这是为了保持词向量的方差，使其适应后续层的训练。
        return self.lut(x) * math.sqrt(self.d_model)


"""
为每个输入位置添加一个唯一的位置编码
这个编码会被添加到词向量中，使模型能够理解位置信息
"""


class PositionalEncoding(nn.Module):
    """
    参数:
        d_model: 词向量的维度
        dropout: dropout概率
        max_len: 句子的最大长度
    位置编码的公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中pos是词在句子中的位置，i是词向量的维度索引
    位置编码矩阵的维度为: max_len × d_model
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个size为 max_len(设定的最大长度)×embedding维度 的全零矩阵
        # 来存放所有小于这个长度位置对应的positional embedding
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 根据公式，计算各个位置在各embedding维度上的位置纹理值，存放到pe矩阵中
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加1个维度，使得pe维度变为：1×max_len×embedding维度
        # (方便后续与一个batch的句子所有词的embedding批量相加)
        pe = pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


"""
这个函数是Transformer模型的基础，它实现了模型的核心注意力机制
"""


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    """
        Encoder：主要使用padding mask处理不等长序列
        Decoder：同时使用padding mask和sequence mask
        padding mask处理填充部分
        sequence mask防止看到未来信息
    """
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 这里是库函数，直接帮你操作好了

    # 将mask后的attention矩阵按照最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


"""
这个MultiHeadedAttention类,实现多头注意力机制
"""


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):  # h是head数量
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0  # 因为要平分到每个head上
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        # self.linears = clones(nn.Linear(d_model, d_model), 4)#不容易理解，改为下面的版本
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = None  # 存放注意力矩阵
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        # query,key,value: 输入的张量，形状通常是 [Batch_Size, Seq_Len, d_model]
        # =================================================

        # 我们要把 d_model  切成 num_heads  * d_k 的形状
        # view: [Batch, Seq_Len, d_model] -> [Batch, Seq_Len, num_heads, d_k]
        # transpose: 交换维度 1 和 2 -> [Batch, num_heads, Seq_Len, d_k]
        # 为什么要交换？为了让 num_heads 靠近 batch 维度，这样 PyTorch 就能并行一次算出所有头的注意力
        # query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #                      for l, x in zip(self.linears, (query, key, value))]
        query = self.w_q(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # =================================================
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        # contiguous()：这是一个内存补丁。在某些情况下，PyTorch中的transpose操作会导致张量在内存中变得不连续。
        # 使用contiguous()方法可以确保张量在内存中是连续的，从而允许后续的view操作正确地重新组织数据。
        # view()：用于重新调整张量的形状。在这里，我们将张量重新调整为 [nbatches, -1, self.h * self.d_k] 的形状，
        # 其中 -1 表示自动计算该维度的大小，以适应总元素数量不变的原则。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        # return self.linears[-1](x)
        return self.w_o(x)


"""
定义一个层归一化的类
"""


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):  # features（特征维度）。
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 这是一个前向传播函数，用于执行Layer Normalization操作
        # 输入x是神经网络层的输出
        # 按最后一个维度计算均值和标准差
        # keepdim=True确保输出的维度与输入相同
        mean = x.mean(-1, keepdim=True)  # 计算最后一个维度的均值
        std = x.std(-1, keepdim=True)  # 计算最后一个维度的标准差

        # 返回Layer Norm的结果
        # Layer Norm公式: y = a * (x - mean) / (std + eps) + b,其实可以变为根号下（方差+eps）
        # 其中a和b是可学习的参数，eps是为了防止除以0的小常数
        return self.gamma * (x - mean) / math.sqrt((std ** 2 + self.eps)) + self.beta


"""
Transformer模型中前馈神经网络的实现。
"""


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将维度从d_model扩展到d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将维度从d_ff压缩回d_model
        self.dropout = nn.Dropout(dropout)  # dropout层，用于防止过拟合

    def forward(self, x):
        return self.w_2(self.dropout(nn.GELU(self.w_1(x))))  # 先通过第一个线性层，然后应用GeLU（进行微调）激活函数，可能会面试


"""
SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起,起到连接的作用
"""


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层
    连在一起只不过每一层输出之后都要先做Layer Norm再残差连接
    func是回调函数数
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # LN层，层归一化，传入特征维度size
        self.dropout = nn.Dropout(dropout)

    """
    sublayer参数是SublayerConnection类中的核心组件，
    它代表了Transformer中的具体处理层，通过这种设计实现了代码的模块化和灵活性。
    """

    def forward(self, x, func):  # 这里的x是输入的张量，在这里是上层输出的qkv三个矩阵的合体
        # 返回Layer Norm和残差连接后结果
        #这里的func就是传入的MultiHeadedAttention或者PositionwiseFeedForward
        return x + self.dropout(func(self.norm(x)))

    """
    我们在代码实现的是这样的结构，刚好和论文的相反
    Input ---> [LayerNorm] ---> [Attention] ---> (+) ---> Output
      |                                           ^
      |___________________________________________|
    """


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 编码器层
class EncoderLayer(nn.Module):
    """
    编码器层包含两个子层:
    1. 多头自注意力机制（Multi-Head Self-Attention）
    2. 前馈神经网络（Position-wise Feed-Forward Network）
    这两个子层都使用了残差连接和层归一化。
    该类的forward方法定义了数据如何通过这两个子层进行处理。
    具体来说:
    - 首先，输入x通过第一个子层，即多头自注意力机制，计算自注意力表示。
    - 然后，结果通过第二个子层，即前馈神经网络，进一步处理。
    - 每个子层的输出都通过SublayerConnection进行残差连接和层归一化。
    该设计使得模型能够有效地捕捉输入序列中的依赖关系，并进行非线性变换，从而提升模型的表达能力。
    该类是Transformer模型中编码器部分的基本构建块。
    参数:
        size: 模型的特征维度
        self_attn: 多头自注意力机制的实例
        feed_forward: 前馈神经网络的实例
        dropout: dropout概率  默认为0.1
    """
    def __init__(self, size, self_attn: attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # SublayerConnection的作用就是把multi和ffn连在一起，其实就是把他俩封装一个块
        # 只不过每一层输出之后都要先做Layer Norm再残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        # sublayer[i](参数1，参数2)，其中参数1是输入的张量，参数2是一个回调函数
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x,
                                                         mask))  # lambda函数传入的x是上层传入的qkv矩阵,返回的是attn的结果然后传参给sublayer[0]
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


# 编码器包含多个编码器层
class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    #如果每一层的主干道 x 都不做归一化，那最后输出的时候 x 岂不是数值很大？”没错！所以 Pre-Norm 架构必须有一个补丁：在整个 Encoder 的最后，必须手动加一个 LayerNorm。


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory
        # 自注意力机制
        # Self-Attention：注意self-attention的q，k和v均为（decoder hidden）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))#解码器的掩码多头注意力机制
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        #                                          解码器查（Q），编码器供（K, V）。
        #   交叉注意力机制                     Attention(查询=解码器, 键=编码器, 值=编码器)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))#交叉注意力机制
        return self.sublayer[2](x, self.feed_forward)#前馈神经网络


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)#1. 数值稳定性（为了不报错）；2. 优化计算速度。


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed# 编码器吃源语言：用 src_embed
        self.tgt_embed = tgt_embed# 解码器吃目标语言：用 tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建完整的Transformer模型
    Args:
        src_vocab: 源语言词汇表大小
        tgt_vocab: 目标语言词汇表大小
        N: 层数
        d_model: 模型维度
        d_ff: 前馈网络维度
        h: 注意力头数
        dropout: dropout率

    Returns:model.to(DEVICE)

    """
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    # 实例化Transformer模型对象

    # 这里的顺序是按照Transformer的init函数来的
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # 初始化模型参数
    # 遍历模型中的所有参数
    for p in model.parameters():
        # 判断参数是否为二维或更高维（例如权重矩阵，而不是偏置向量）
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)
