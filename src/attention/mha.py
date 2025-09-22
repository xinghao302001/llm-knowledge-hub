"""
多头自注意力
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        bias: bool = True,
    ):
        """
        初始化多头自注意力

        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout率
            temperature: 注意力温度
            bias: 是否使用偏置
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model  # embedding_size
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # head_dim
        self.drop_out = dropout
        self.bias = bias

        # 线性投影层
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=self.bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # drop out
        self.drop_out = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重
        采用Xavier均匀分布初始化线性层权重，偏置初始化为0
        """
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def attetion_score(
        Q, K, V, mask=None, dropout=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注意力分数

        Args:
            Q: 查询张量 [batch_size, seq_len, d_k]
            K: 键张量 [batch_size, seq_len, d_k]
            V: 值张量 [batch_size, seq_len, d_v]
            mask: 掩码张量 [batch_size, seq_len, seq_len]
            dropout: dropout层
        Returns:
            output: 输出张量 [batch_size, seq_len, d_v]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # 处理掩码
        if mask is not None:
            pass  # TODO
        p_attn = torch.softmax(scores, dim=-1)  # 计算注意力权重

        # 应用dropout
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, V), p_attn

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            query: 查询张量 [batch_size, seq_len, d_model]
            key: 键张量 [batch_size, seq_len, d_model]
            value: 值张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len, seq_len_k] 或 [batch_size, n_heads, seq_len_q, seq_len_k]
            return_attention: 是否返回注意力权重

        Returns:
            output: 输出张量 [batch_size, seq_len_q, d_model]
            attention_weights: 注意力权重(可选) [batch_size, n_heads, seq_len_q, seq_len_k]
        """

        batch_size, seq_len, d_model = query.size()

        # 1. 线性投影: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_heads * d_k]
        Q = self.w_q(query)
        K = self.w_k(keys)
        V = self.w_v(values)

        # # 2. 构造多头: [batch_size, seq_len, n_heads * d_k] -> [batch_size, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. 处理掩码
        if mask is not None:
            pass  # TODO

        # 4. 计算每个头的注意力: [batch_size, n_heads, seq_len, d_head], [batch_size, n_heads, seq_len, seq_len]
        attention_output, attention_weights = self.attetion_score(
            Q, K, V, mask=mask, dropout=self.drop_out
        )

        # 5. 合并多头输出: [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads * d_k]
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # 6. 最终线性投影
        output = self.w_o(attention_output)

        # 7. 返回结果
        if return_attention:
            return output, attention_weights
        else:
            return output, None


##  test code
if __name__ == "__main__":
    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2

    mha = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)

    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)

    output, attn_weights = mha(query, key, value, return_attention=True)

    print("Output shape:", output.shape)  # Expected: [2, 4, 8]
    if attn_weights is not None:
        print("Attention weights shape:", attn_weights.shape)  # Expected: [2, 2, 4, 4]
