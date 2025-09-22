from typing import Optional, Tuple
import math

import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):
    """多查询注意力机制"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.bias = bias

        self.w_q = nn.Linear(d_model, d_model, bias=self.bias)
        self.w_k = nn.Linear(d_model, self.d_head, bias=self.bias)  # [MQA]
        self.w_v = nn.Linear(d_model, self.d_head, bias=self.bias)  # [MQA]
        self.w_o = nn.Linear(d_model, d_model, bias=self.bias)

        self.dropout = nn.Dropout(dropout)

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
            Q: [B, H, len_q, d_head]
            K: [B, 1, len_k, d_head]   # [MQA] 广播到 head 维
            V: [B, 1, len_v, d_head]   # [MQA] 广播到 head 维
            mask: #TODO
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # [MQA]
        if mask is not None:
            pass
        p_attn = torch.softmax(scores, dim=-1)
        if dropout:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, V), p_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query: (batch, seq_len, embed_dim)
        key:   (batch, seq_len, embed_dim)
        value: (batch, seq_len, embed_dim)
        mask:  (batch, 1, seq_len) or (batch, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = query.size()

        # 1) 线性转换
        Q = self.w_q(query)
        ## [MQA] K/V 仅投影到 d_head，并且不再分多头
        K = self.w_k(key)
        V = self.w_v(value)

        # 2) 构造多头 Q；K/V 在 head 维共享（通过 unsqueeze 到 head 维再广播）
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.unsqueeze(1)  # [B,1,L_k,d_head]  # [MQA]
        V = V.unsqueeze(1)  # [B,1,L_k,d_head]  # [MQA]

        # 3) 处理掩码
        if mask is not None:
            pass

        # 4) 注意力分数  # out:[B,H,L_q,d_head], attn:[B,H,L_q,L_k]
        attention_output, attention_weights = self.attetion_score(
            Q, K, V, mask=mask, dropout=self.dropout
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
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    d_model = 8
    n_heads = 2

    mha_mqa = MultiQueryAttention(d_model=d_model, n_heads=n_heads)

    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)

    # 构造一个简单的全1 mask（无屏蔽）
    mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool)

    output, attn_weights = mha_mqa(query, key, value, mask=mask, return_attention=True)

    print("Output shape:", output.shape)  # [2, 4, 8]
    if attn_weights is not None:
        print("Attention weights shape:", attn_weights.shape)  # [2, 2, 4, 4]
