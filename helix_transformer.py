# import torch, torch.nn as nn
# from .helix_encoder import HelixPosEncoder

# class SpiralMatchBlock(nn.Module):
#     def __init__(self, d):
#         super().__init__()
#         self.proj = nn.Linear(d, d)
#         self.mse  = nn.MSELoss(reduction='none')

#     def forward(self, h_v, h_a, h_t):
#         # center = (h_v + h_a) / 2            # perceptual center

#         # → 允许 V/A 权重不同 & 引入“叉乘”做方向性
#         # alpha = 0.6 # 可调或可学
#         # center = alpha * h_v + (1-alpha) * h_a<br># 叉乘 / 斜率特征（粗糙写法）
#         # wave = torch.cross(h_v, h_a, dim=-1) # (B,T,d)

#         # α 可学习，允许 V/A 权重自动搜索
#         if not hasattr(self, "alpha"):
#             self.alpha = nn.Parameter(torch.tensor(0.6))
#         center = self.alpha * h_v + (1 - self.alpha) * h_a

#         # 方向性特征：若维度 <3 则退化为 center
#         wave = torch.cross(h_v[..., :3], h_a[..., :3], dim=-1) \
#                if h_v.size(-1) >= 3 else center


#         trust  = self.proj(h_t)
#         err    = self.mse(center, trust).mean(-1, keepdim=True)  # (B,T,1)
#         fusion = torch.cat([center, trust, err], dim=-1)         # (B,T,2d+1)
#         return fusion, err                  # err 供损失 & 可视化

# class HelixTransformer(nn.Module):
#     def __init__(self, feat_dim, d_model=384, n_heads=6,
#                  n_layers=2, dropout=0.1, max_len=1024):
#         super().__init__()
#         # self.embed = nn.Linear(feat_dim, d_model)
#         # self.pos   = HelixPosEncoder()
#         # self.pos_proj = nn.Linear(3, d_model, bias=False)   
#         # self.chunk_dim = d_model // 3        # 192 // 3 = 64



#         # enc_layer  = nn.TransformerEncoderLayer(
#         #     d_model, n_heads, d_model*4, dropout, batch_first=True)
#         # self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
#         # self.match = SpiralMatchBlock(self.chunk_dim)
#         # self.cls   = nn.Linear(self.chunk_dim * 2 + 1, 1)

#         self.embed = nn.Linear(feat_dim, d_model)
#         self.pos   = HelixPosEncoder()
#         self.pos_proj = nn.Linear(3, d_model, bias=False)

#         # --- Transformer Encoder ---
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model, n_heads, d_model * 4, dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

#         # --- 3 路投影 + SpiralMatch ---
#         self.chunk_dim = d_model // 4               # 384//4 = 96
#         self.v_proj = nn.Linear(d_model, self.chunk_dim, bias=False)
#         self.a_proj = nn.Linear(d_model, self.chunk_dim, bias=False)
#         self.t_proj = nn.Linear(d_model, self.chunk_dim, bias=False)

#         self.match = SpiralMatchBlock(self.chunk_dim)
#         self.cls   = nn.Linear(self.chunk_dim * 2 + 1, 1)



#     def forward(self, x):                   # x (B,T,F)
#         B, T, _ = x.size()
#         t_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
#         pos   = self.pos_proj(self.pos(t_idx))   # 3 维 → 64 维
#         # h     = self.embed(x) + pos              # 64
#         # h = self.encoder(h)                 # (B,T,d)

#         # # imagined: [valence, arousal, trustworthiness, other features...]
#         # # as long as top3 are there then we have chunk；add other features if needed they can join regular attention
#         # # h_v, h_a, h_t, h_rest = torch.split(h, [1,1,1,h.size(-1)-3], dim=-1)
#         # h_v, h_a, h_t = torch.chunk(h, 3, dim=-1)    # 64 dim each 
#         # fusion, err   = self.match(h_v, h_a, h_t)     # no more squeeze
#         # # fusion, err = self.match(h_v.squeeze(-1), h_a.squeeze(-1), h_t.squeeze(-1))
#         # logits = self.cls(fusion)           # (B,T,1)

#         h     = self.embed(x) + pos              # (B,T,d_model)
#         h     = self.encoder(h)                  # (B,T,d_model)

#         # 三路投影 → chunk_dim
#         h_v = self.v_proj(h)                     # (B,T,chunk)
#         h_a = self.a_proj(h)
#         h_t = self.t_proj(h)

#         fusion, err = self.match(h_v, h_a, h_t)  # (B,T,2*chunk+1)
#         logits = self.cls(fusion)                # (B,T,1)
#         return logits, err                  # err 参与损失




import torch
import torch.nn as nn
from .helix_encoder import HelixPosEncoder


class SpiralMatchBlock(nn.Module):
    """
    输入三路 hidden (V, A, T) —> 融合 + 两级误差
    输出:
        fusion  (B,T, dim_f)
        err     (B,T, 1)   = err_align + 0.3 * err_circ
    """
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        self.mse  = nn.MSELoss(reduction="none")
        self.alpha = nn.Parameter(torch.tensor(0.3))     # 可学习权重

    def forward(self, h_v, h_a, h_t):
        # ── 中心 + 波形 ──────────────────────────────
        center = self.alpha * h_v + (1.0 - self.alpha) * h_a       # (B,T,d)

        # 若 d >= 3 取前三维做叉乘，否则退化为 0 向量
        if h_v.size(-1) >= 3:
            wave = torch.cross(h_v[..., :3], h_a[..., :3], dim=-1)  # (B,T,3)
        else:
            wave = torch.zeros_like(center)

        # ── Trust 投影 & 误差 ────────────────────────
        trust = self.proj(h_t)                                     # (B,T,d)

        # ① 欧氏对齐误差
        err_align = self.mse(center, trust).mean(-1, keepdim=True)     # (B,T,1)

        # ② 方向一致性误差（单位化后再算 MSE）
        v1 = center / (center.norm(dim=-1, keepdim=True) + 1e-6)
        v2 = trust  / (trust .norm(dim=-1, keepdim=True) + 1e-6)
        err_circ = self.mse(v1, v2).mean(-1, keepdim=True)             # (B,T,1)

#        err = err_align + 0.3 * err_circ

        self.gamma = getattr(self, "gamma", nn.Parameter(torch.tensor(0.3)))
        err = err_align + self.gamma.abs() * err_circ

        # fusion = center | trust | wave | err
        fusion = torch.cat([center, trust, wave, err], dim=-1)
        return fusion, err


class HelixTransformer(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 d_model: int = 384,
                 n_heads: int = 6,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()

        # ── 输入嵌入 + 螺旋位置 ───────────────────────
        self.embed     = nn.Linear(feat_dim, d_model)
        self.pos_enc   = HelixPosEncoder()
        self.pos_proj  = nn.Linear(3, d_model, bias=False)

        # ── Transformer 编码器 ──────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── 三路投影 + 匹配 ─────────────────────────
        self.chunk_dim = d_model // 4                    # 96
        self.v_proj = nn.Linear(d_model, self.chunk_dim, bias=False)
        self.a_proj = nn.Linear(d_model, self.chunk_dim, bias=False)
        self.t_proj = nn.Linear(d_model, self.chunk_dim, bias=False)

        self.match  = SpiralMatchBlock(self.chunk_dim)

        # fusion:   center (c) + trust (c) + wave (3) + err (1)
        self.cls_in = self.chunk_dim * 2 + 3 + 1
        self.cls    = nn.Linear(self.cls_in, 1)

    # ----------------------------------------------------------

    def forward(self, x):                # x (B,T,F)
        B, T, _ = x.shape
        t_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        pos   = self.pos_proj(self.pos_enc(t_idx))

        h = self.embed(x) + pos          # (B,T,d_model)
        h = self.encoder(h)              # (B,T,d_model)

        # 三路投影
        h_v = self.v_proj(h)
        h_a = self.a_proj(h)
        h_t = self.t_proj(h)

        # fusion, err = self.match(h_v, h_a, h_t)      # (B,T, cls_in)
        # logits = self.cls(fusion)                    # (B,T,1)
        # return logits, err
        fusion, err_micro = self.match(h_v, h_a, h_t)   # (B,T,…)
        err_macro = err_micro.mean(dim=1, keepdim=True) # (B,1,1)
        logits = self.cls(fusion)
        return logits, (err_micro, err_macro)