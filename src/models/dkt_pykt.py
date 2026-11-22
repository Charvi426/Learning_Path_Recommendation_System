"""Optional deep baseline (METHODOLOGY 5.4): DKT from pyKT on topic sequences.

One sequence per student-presentation; concept = module-topic id; response = y (submitted topics only, in
topic order). pyKT's DKT model class is used with its default size (emb_size = 200, dropout = 0.1) and a plain
Adam training loop (lr 1e-3, early stopping on validation AUC), because pyKT's own training scripts expect its
preprocessed benchmark folders and wandb.

The prediction for step i uses only responses at steps < i. The first topic of a sequence has no history, so DKT
is evaluated on rows with >= 1 earlier response, and the other models are re-scored on the same rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sequences(inter: pd.DataFrame, tindex: dict, split: str):
    d = inter[(inter["split"] == split) & inter["y"].notna()].sort_values(["sid", "topic"])
    seqs = []
    for sid, g in d.groupby("sid"):
        q = [tindex[(m, int(k))] for m, k in zip(g["code_module"], g["topic"])]
        seqs.append((sid, np.array(q), g["y"].to_numpy(int), g["topic"].to_numpy()))
    return seqs


def _batch(seqs, maxlen):
    B = len(seqs)
    Q = np.zeros((B, maxlen), np.int64)
    R = np.zeros((B, maxlen), np.int64)
    Mk = np.zeros((B, maxlen), bool)
    for i, (_, q, r, _) in enumerate(seqs):
        n = len(q)
        Q[i, :n], R[i, :n], Mk[i, :n] = q, r, True
    return Q, R, Mk


def train_dkt(inter, tindex, seed=42, epochs=60, batch=64, lr=1e-3, patience=8):
    import torch
    from pykt.models.dkt import DKT
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    num_c = len(tindex)
    tr, va = sequences(inter, tindex, "train"), sequences(inter, tindex, "val")
    maxlen = max(len(s[1]) for s in tr + va)
    model = DKT(num_c=num_c, emb_size=200, dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(seqs):
        Q, R, Mk = (torch.tensor(a) for a in _batch(seqs, maxlen))
        y = model(Q, R)                                 # (B, L, num_c): prediction for the NEXT step
        pred = (y[:, :-1, :] * torch.nn.functional.one_hot(Q[:, 1:], num_c)).sum(-1)
        tgt = R[:, 1:].float()
        m = Mk[:, 1:]
        return pred[m], tgt[m]

    best, best_state, bad = -1, None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), batch):
            p, t = run([tr[j] for j in order[i:i + batch]])
            loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p, t = run(va)
        auc = roc_auc_score(t.numpy(), p.numpy())
        if auc > best + 1e-4:
            best, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, maxlen, {"best_val_auc": best, "epochs_run": ep + 1}


def predict_dkt(model, maxlen, inter, tindex, split):
    """Returns DataFrame sid, topic, p for steps >= 2 of each sequence."""
    import torch
    seqs = sequences(inter, tindex, split)
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            chunk = seqs[i:i + 256]
            Q, R, Mk = (torch.tensor(a) for a in _batch(chunk, maxlen))
            y = model(Q, R)
            pred = (y[:, :-1, :] * torch.nn.functional.one_hot(Q[:, 1:], len(tindex))).sum(-1).numpy()
            for b, (sid, q, r, topics) in enumerate(chunk):
                for step in range(1, len(q)):
                    out.append((sid, int(topics[step]), float(pred[b, step - 1])))
    return pd.DataFrame(out, columns=["sid", "topic", "p_dkt"])
