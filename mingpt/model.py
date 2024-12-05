"""
GPT语言模型的完整定义，全部在这个单一文件中。

参考资料：
1) OpenAI发布的GPT-2 TensorFlow官方实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    GELU激活函数的实现，目前在Google BERT仓库中（与OpenAI GPT相同）。
    参考：Gaussian Error Linear Units (GELU) 论文：https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    一个普通的多头掩码自注意力层，最后有一个投影。
    这里可以使用torch.nn.MultiheadAttention，但我包含了一个显式实现，以表明这里没有什么可怕的。
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 为所有头部计算键、查询、值的投影，但在一个批次中
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码，确保注意力只应用于输入序列的左侧
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # 批次大小，序列长度，嵌入维度（n_embd）

        # 为批次中的所有头部计算查询、键、值，并将头部前移为批次维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 因果自注意力；自注意力：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 重新组合所有头部输出

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ 一个不起眼的Transformer块 """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP前向传播

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT语言模型 """

    @staticmethod
    def get_default_config():
        C = CN()
        # 配置中必须给出model_type或(n_layer, n_head, n_embd)之一
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # 这些选项必须在外部填写
        C.vocab_size = None
        C.block_size = None
        # dropout超参数
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # 这两者之一必须给出（XOR）
        if type_given:
            # 从model_type翻译到详细配置
            config.merge_from_dict({
                # 名称遵循huggingface命名约定
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M参数
                # GPT-2配置
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # （还有更多...）
                # 我编造了这些小模型
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有权重，并对残差投影应用特殊的缩放初始化，参考GPT-2论文
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量（注意我们不计算lm_head中的解码器参数）
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("参数数量: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        通过复制huggingface/transformers检查点的权重来初始化预训练的GPT模型。
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # 创建一个从头初始化的minGPT模型
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai的模型词汇表
        config.block_size = 1024  # openai的模型块大小
        model = GPT(config)
        sd = model.state_dict()

        # 初始化一个huggingface/transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制时确保所有参数对齐并匹配名称和形状
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # 忽略这些
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上openai检查点使用了一个"Conv1D"模块，但我们只想使用一个普通的nn.Linear。
        # 这意味着在导入它们时我们必须转置这些权重
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的Conv1D权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        这个长函数实际上做了非常简单的事情，并且非常防御性：
        我们将模型的所有参数分为两类：那些将经历权重衰减以进行正则化的参数和那些不会的参数（偏置和层归一化/嵌入权重）。
        然后我们返回PyTorch优化器对象。
        """

        # 将所有参数分为将经历正则化权重衰减的和不会的
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 完整参数名称
                # 随机注释：由于named_modules和named_parameters是递归的
                # 我们会多次看到相同的张量p。但这样做的好处是
                # 我们可以知道任何张量p属于哪个父模块...
                if pn.endswith('bias'):
                    # 所有偏置都不会衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重不会衰减
                    no_decay.add(fpn)

        # 验证我们考虑了每个参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数%s同时出现在衰减/不衰减集合中！" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数%s未被分为衰减/不衰减集合之一！" \
                                                    % (str(param_dict.keys() - union_params), )

        # 创建pytorch优化器对象
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"无法前向传播长度为{t}的序列，块大小仅为{self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # 形状 (1, t)

        # 前向传播GPT模型本身
        tok_emb = self.transformer.wte(idx) # 形状为(b, t, n_embd)的token嵌入
        pos_emb = self.transformer.wpe(pos) # 形状为(1, t, n_embd)的位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 如果给定了一些目标，也计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        获取一个索引序列idx（形状为(b,t)的LongTensor），并完成
        序列max_new_tokens次，每次将预测结果反馈到模型中。
        大多数情况下，你会希望确保处于model.eval()模式下操作。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，我们必须在block_size处裁剪它
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 前向传播模型以获取序列中索引的logits
            logits, _ = self(idx_cond)
            # 提取最后一步的logits并按所需温度缩放
            logits = logits[:, -1, :] / temperature
            # 可选地将logits裁剪为仅前top k选项
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用softmax将logits转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样或选择最可能的元素
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # 将采样的索引附加到运行序列并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx