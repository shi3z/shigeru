"""Phase B9-A question 4 — is the FINAL block predictable from intermediate hidden states beyond what the LM head says?
(docs/global_block_denoising.md §2.4)

Data: results/pure/traj_s16_left_tf_hidden.pt (bench/diag_trajectory.py --save-hidden): for every step and every still-
masked position, the final-layer hidden state h, the step's top-8 candidates with their LM-head log-probs, and the
FINAL token of the run.  Task: among the step's top-8 candidates, pick the FINAL token.
  LM head        rank by the step's own log-prob (what the sampler does: top-1 recall of FINAL)
  linear probe   score_c = (W h) . E[c] + b . lp_c  — a low-rank bilinear re-scorer over the candidate embeddings E
                 (frozen input embeddings) plus the LM-head log-prob, trained with cross-entropy on blocks of prompts
                 0–24 and evaluated on prompts 25–49 (positions whose FINAL token is among the top-8 only; the
                 containment rate is reported separately).
If the probe beats the LM head by a wide margin, the hidden state carries information about the eventual block that
the current argmax discards; if not, the LM head already extracts what the state knows.
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv-train/bin/python bench/probe_hidden.py --data results/pure/traj_s16_left_tf_hidden.pt \
      --model runs/zeta_sft_final_e5/final --out results/pure/probe_hidden_s16.json"""
import argparse, collections, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.gpu_guard import check_env
check_env()
import torch
from transformers import AutoModelForCausalLM
from bench.common import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    d = torch.load(a.data)
    blocks = d["blocks"]
    ids = sorted(set(b["id"] for b in blocks))
    train_ids = set(ids[: len(ids) // 2])
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    E = lm.get_input_embeddings().weight.detach().float().to(dev)      # [V, H]
    del lm
    rows = collections.defaultdict(list)                                 # split -> (h, cand[8], lp[8], label idx, step)
    contain = collections.Counter(); total = collections.Counter()
    for b in blocks:
        split = "train" if b["id"] in train_ids else "test"
        final = torch.tensor(b["final"])
        for st in b["steps"]:
            m = st["mask"]
            for q in m.nonzero().flatten().tolist():
                cands = st["topk_ids"][q]
                total[(split, st["step"])] += 1
                hit = (cands == final[q]).nonzero().flatten()
                if len(hit):
                    contain[(split, st["step"])] += 1
                    rows[split].append((st["hidden"][q], cands, st["topk_lp"][q], int(hit[0]), st["step"]))
    def pack(rs):
        H = torch.stack([r[0] for r in rs]).float().to(dev); C = torch.stack([r[1] for r in rs]).to(dev); L = torch.stack([r[2] for r in rs]).float().to(dev)
        Y = torch.tensor([r[3] for r in rs], device=dev); S = torch.tensor([r[4] for r in rs], device=dev)
        return H, C, L, Y, S
    Htr, Ctr, Ltr, Ytr, Str = pack(rows["train"]); Hte, Cte, Lte, Yte, Ste = pack(rows["test"])
    hid = Htr.shape[1]
    W = torch.nn.Linear(hid, a.rank, bias=False).to(dev); U = torch.nn.Linear(E.shape[1], a.rank, bias=False).to(dev)
    beta = torch.nn.Parameter(torch.ones((), device=dev)); gamma_ = torch.nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.Adam(list(W.parameters()) + list(U.parameters()) + [beta, gamma_], lr=1e-3, weight_decay=1e-4)

    def scores(H, C, L):
        hc = W(H)                                   # [N, r]
        ec = U(E[C])                                # [N, 8, r]
        return (ec * hc[:, None]).sum(-1) / hid ** 0.5 * torch.exp(gamma_) + beta * L

    bs = 4096
    for ep in range(a.epochs):
        perm = torch.randperm(len(Ytr), device=dev)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.cross_entropy(scores(Htr[idx], Ctr[idx], Ltr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        res = {}
        for name, (H, C, L, Y, S) in (("train", (Htr, Ctr, Ltr, Ytr, Str)), ("test", (Hte, Cte, Lte, Yte, Ste))):
            lm_top1 = (L.argmax(-1) == Y).float(); pr_top1 = (scores(H, C, L).argmax(-1) == Y).float()
            by_step = {}
            for s in sorted(set(S.tolist())):
                m = S == s
                n_tot = total[(name, s)]; n_in = contain[(name, s)]
                by_step[str(s)] = dict(n=int(m.sum()), containment_top8=round(n_in / max(1, n_tot), 4), lm_head_top1_given_contained=round(float(lm_top1[m].mean()), 4),
                                       probe_top1_given_contained=round(float(pr_top1[m].mean()), 4),
                                       lm_head_recall_final=round(float(lm_top1[m].sum()) / max(1, n_tot), 4), probe_recall_final=round(float(pr_top1[m].sum()) / max(1, n_tot), 4))
            res[name] = dict(n=len(Y), lm_head_top1_given_contained=round(float(lm_top1.mean()), 4), probe_top1_given_contained=round(float(pr_top1.mean()), 4), by_step=by_step)
    print("=== probe", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_step"} for k, v in res.items()}), flush=True)
    print("=== probe-test-by-step", json.dumps(res["test"]["by_step"]), flush=True)
    save_json(a.out, dict(data=a.data, rank=a.rank, epochs=a.epochs, train_prompts=sorted(train_ids), results=res))


if __name__ == "__main__":
    main()
