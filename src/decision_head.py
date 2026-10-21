"""JLT-Head: Jev-style long-tail decision head.

Design (see docs/design.md):
  context tokens   : encoder feature-map tokens, full attention among themselves
  candidate tokens : prototypes + prior metadata; attend to context + self only
  question tokens  : Q1 coarse / Q2 fine / Q4 cover; attend to context + self only
  readouts (pointer, kev-style):
    Q2 fine   : h_q2 vs candidate tokens -> logits over K classes + abstain
                with prior-conditioned logit adjustment (lambda * log pi)
    Q3 verify : per-candidate binary head on candidate tokens (balanced task)
    Q1 coarse : h_q1 vs group tokens (group token = mean of member candidate h)
    Q4 cover  : binary head on q4 token (OOD gate)
  Abstention is a first-class learned candidate; it is excluded from the Q2
  softmax unless out-of-distribution negatives are supplied at training time.

The candidate table is an INPUT: prototypes (K, d_enc) + log-priors (K,) +
group ids (K,) may change at test time (class addition) with zero retraining.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    def __init__(self, d=256, heads=8, ffn_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class JLTHead(nn.Module):
    def __init__(self, d_enc=64, d=256, layers=4, heads=8, n_question_types=4):
        super().__init__()
        self.d = d
        self.ctx_proj = nn.Linear(d_enc, d)
        self.proto_proj = nn.Linear(d_enc, d)
        self.prior_proj = nn.Linear(1, d)
        self.type_emb = nn.Embedding(2, d)               # 0 candidate, 1 question
        self.qtype_emb = nn.Embedding(n_question_types, d)  # q1, q2, q3(proxy), q4
        self.abstain_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.abstain_token, std=0.02)

        self.blocks = nn.ModuleList(TransformerLayer(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)

        # pointer readout (shared q/k projections for Q2 and Q1)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))
        self.verify_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.cover_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # prior-conditioned logit adjustment strength (0 = off)
        self.lam = nn.Parameter(torch.zeros(1))
        # residual NCM branch: logits = alpha*cos(pooled, proto) + beta*ptr.
        # alpha=1, beta=0 at init == NCM-cos (training-free floor), so the
        # transformer pointer can only be learned ON TOP of that floor.
        self.ncm_scale = nn.Parameter(torch.tensor(20.0))
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def _ptr_logits(self, h_q, h_k):
        """Cosine pointer with learnable temperature (NCM-cos parity)."""
        q = F.normalize(self.ptr_q(h_q), dim=-1)
        k = F.normalize(self.ptr_k(h_k), dim=-1)
        scale = self.logit_scale.clamp(1.0, 100.0).exp()
        if h_q.dim() == h_k.dim():
            return scale * (q @ k.transpose(-1, -2))
        return scale * torch.einsum("bd,bkd->bk", q, k)

    def _build_mask(self, M, K, device):
        # layout: [Z: M | C: K+1 | Q: 3(q1,q2,q4)]
        # candidates are CONTEXT-INDEPENDENT anchors (self only) so the head
        # cannot shortcut "look at the image and declare itself"; all image
        # evidence flows through the question tokens (two-tower pointer).
        L = M + K + 1 + 3
        allowed = torch.zeros(L, L, dtype=torch.bool, device=device)
        z = slice(0, M)
        c0, c1 = M, M + K + 1
        q0 = c1
        allowed[z, z] = True                              # context sees context
        eye = torch.eye(L, dtype=torch.bool, device=device)
        allowed[q0:L, z] = True                           # questions see context
        allowed = allowed | eye                           # + self
        m = torch.zeros(L, L, device=device)
        m[~allowed] = float("-inf")
        return m

    def forward(self, ctx, pooled, protos, log_priors, group_ids=None,
                use_abstain=True):
        """ctx: (B, M, d_enc)  pooled: (B, d_enc)  protos: (K, d_enc)
        log_priors: (K,)  group_ids: (K,) int or None (disables Q1)."""
        B, M, _ = ctx.shape
        K = protos.shape[0]
        device = ctx.device

        lp = log_priors.to(device).float()
        lp_n = (lp - lp.mean()) / (lp.std() + 1e-6)

        z = self.ctx_proj(ctx)
        cand = (self.proto_proj(protos) + self.prior_proj(lp_n.unsqueeze(1))
                + self.type_emb.weight[0]).unsqueeze(0).expand(B, -1, -1)
        abst = self.abstain_token.unsqueeze(0)
        q1t = self.qtype_emb.weight[1] + self.type_emb.weight[1]
        q2t = self.qtype_emb.weight[2] + self.type_emb.weight[1]
        q4t = self.qtype_emb.weight[3] + self.type_emb.weight[1]

        one = lambda v: v.view(1, 1, -1).expand(B, 1, self.d)
        seq = torch.cat([z, cand, one(self.abstain_token), one(q1t),
                         one(q2t), one(q4t)], dim=1)
        mask = self._build_mask(M, K, device)
        for blk in self.blocks:
            seq = blk(seq, mask)
        seq = self.ln_f(seq)

        h_cand = seq[:, M:M + K]                          # B,K,d
        h_abst = seq[:, M + K]                            # B,d
        h_q1 = seq[:, M + K + 1]
        h_q2 = seq[:, M + K + 2]
        h_q4 = seq[:, M + K + 3]

        out = {}
        # Q2 fine: residual head = alpha * NCM-cos + beta * cosine pointer
        qn = F.normalize(pooled, dim=-1)                  # B,d_enc
        kn = F.normalize(protos, dim=-1)                  # K,d_enc
        ncm = self.ncm_scale * (qn @ kn.T)                # B,K
        ptr = self._ptr_logits(h_q2, h_cand)              # B,K
        prior = self.lam * lp.unsqueeze(0)                # B,K
        logits = self.alpha * ncm + self.beta * ptr + prior
        # evidence gating: score = ncm_part + gamma_i*ptr_part + prior_part
        # (prior applies uniformly; only the LEARNED pointer is gated)
        out["ncm_part"] = self.alpha * ncm
        out["ptr_part"] = self.beta * ptr
        out["prior_part"] = prior.expand(B, -1)
        if use_abstain:
            abst_logit = self._ptr_logits(h_q2, h_abst.unsqueeze(1))  # B,1
            out["logits_q2"] = torch.cat([logits, abst_logit], dim=1)
        else:
            out["logits_q2"] = logits
        # Q3 verify: anchor-image interaction (elementwise product -> MLP)
        vf = h_cand * h_q2.unsqueeze(1)                       # B,K,d
        out["verify"] = self.verify_head(vf).squeeze(-1)      # B,K
        # Q4 cover
        out["cover"] = self.cover_head(h_q4).squeeze(-1)       # B
        # Q1 coarse: cosine pointer over group tokens
        if group_ids is not None:
            g = group_ids.to(device)
            G = int(g.max()) + 1
            gtok = torch.zeros(G, self.d, device=device)
            gtok.index_add_(0, g, h_cand.mean(0))
            cnt = torch.bincount(g, minlength=G).clamp(min=1).unsqueeze(1)
            gtok = gtok / cnt
            out["logits_q1"] = self._ptr_logits(h_q1, gtok)  # B,G
            out["group_ids"] = g
        return out


class MultiQuestionLoss(nn.Module):
    """Sum of per-question CE/BCE terms, each normalized by a running RMS estimate
    (DreamerV4-style loss normalization so weights are comparable in nats)."""

    def __init__(self, w=dict(q1=0.1, q2=1.0, q3=0.5, q4=0.25), beta=0.999):
        super().__init__()
        self.w = w
        self.beta = beta
        self.register_buffer("rms", torch.ones(len(w)))
        self.names = ["q1", "q2", "q3", "q4"]

    def forward(self, out, target_fine, target_group=None):
        """target_fine: (B,) in [0, K], value K means abstain (requires logits_q2
        to have been produced with use_abstain=True). Abstain-target samples
        are OOD negatives: no candidate is correct (Q3 target all-zeros) and
        the candidate table does not cover them (Q4 target 0)."""
        B, K = out["verify"].shape
        losses = {}
        logits_q2 = out["logits_q2"]
        assert logits_q2.shape[1] == K or logits_q2.shape[1] == K + 1
        if (target_fine == K).any():
            assert logits_q2.shape[1] == K + 1, "abstain targets need use_abstain=True"
        losses["q2"] = F.cross_entropy(logits_q2, target_fine,
                                       label_smoothing=0.0)
        is_ood = (target_fine == K)
        tgt = F.one_hot(target_fine.clamp(max=K - 1), K).float()
        tgt[is_ood] = 0.0
        # positives are 1-of-K -> weight them K so pos/neg mass is balanced
        losses["q3"] = F.binary_cross_entropy_with_logits(
            out["verify"], tgt, reduction="mean", pos_weight=torch.tensor(
                float(K), device=out["verify"].device))
        losses["q4"] = F.binary_cross_entropy_with_logits(
            out["cover"], (~is_ood).float().to(out["cover"].device))
        if target_group is not None and "logits_q1" in out:
            losses["q1"] = F.cross_entropy(out["logits_q1"], target_group,
                                           ignore_index=-100)
        else:
            losses["q1"] = torch.zeros((), device=out["cover"].device)

        vals = torch.stack([losses[n].detach() for n in self.names])
        with torch.no_grad():
            self.rms.mul_(self.beta).add_(vals.pow(2), alpha=1 - self.beta).clamp_(min=1e-8)
        total = sum(self.w[n] * losses[n] / self.rms[i].sqrt()
                    for i, n in enumerate(self.names))
        return total, {n: float(losses[n].detach()) for n in self.names}
