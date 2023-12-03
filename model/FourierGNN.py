import torch
import torch.nn as nn
import torch.nn.functional as F

class FGN(nn.Module):
    def __init__(self, pre_length, embed_size,
                 feature_size, seq_length, hidden_size, hard_thresholding_fraction=1, hidden_size_factor=1, sparsity_threshold=0.01):
        '''
        Args:
            pre_length:                     预测长度
            embed_size:                     嵌入维度
            feature_size:                   特征数
            seq_length:                     序列长度
            hidden_size:                    隐藏层的大小
            hard_thresholding_fraction:     硬阈值化的比例
            hidden_size_factor:             隐藏层大小的因子
            sparsity_threshold:             稀疏性阈值
        '''
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        # 一个可学习的嵌入矩阵，形状为 (1, embed_size)
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        # 线性层和参数
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))

        # 定义额外的嵌入层 embeddings_10
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))

        # 定义前馈神经网络
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )
        self.to('cuda:0')

    def tokenEmb(self, x):
        '''

        Args:
            x: (batch_size, sequence_length)
            x.unsqueeze(2): (batch_size, sequence_length, 1)

        Returns:
            x * y   --> (batch_size, sequence_length, 1) * (1, embed_size)
                    ---> (batch_size, sequence_length, embed_size)

        '''
        x = x.unsqueeze(2) # 为了与 self.embeddings 的形状相匹配，在第三个维度上增加一个维度
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        o1_real = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N*L)//2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        # ---------------------------------------------------------------------------------------------------------------

        # 第一层傅立叶图卷积操作
        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold) # 对第一层的输出进行软阈值化

        # ---------------------------------------------------------------------------------------------------------------

        # 第二层傅立叶图卷积操作
        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold) # 对第二层的输出进行软阈值化
        x = x + y  # 残差连接

        #---------------------------------------------------------------------------------------------------------------

        # 第三层傅立叶图卷积操作
        o3_real = F.relu(
                torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
                torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
                self.b3[0]
        )

        o3_imag = F.relu(
                torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
                torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
                self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold) # 对第三层的输出进行软阈值化
        z = z + x  # 残差连接

        # ---------------------------------------------------------------------------------------------------------------

        z = torch.view_as_complex(z)  # 回到复数域
        return z

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()  # 将输入张量 x 的维度进行调整
        B, N, L = x.shape
        x = x.reshape(B, -1) # B*N*L ==> B*NL
        x = self.tokenEmb(x)  # embedding B*NL ==> B*NL*D

        # 对嵌入数据在 dim=1 上进行（离散）傅立叶变换
        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        # 保存傅立叶变换的中间结果
        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        # 残差连接
        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # 对嵌入数据在 dim=1 上进行（离散）傅立叶逆变换
        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x

