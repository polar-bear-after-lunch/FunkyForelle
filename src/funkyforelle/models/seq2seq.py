"""Encoder–decoder LSTM for multi-day DO / temperature forecasts.

Carried over from AareML (``src/model.py``, MIT licence, same authors) and
simplified for FunkyForelle:

* any number of input features (so new data sources can be added freely),
* horizon is a constructor argument instead of a global,
* the same AdamW + ReduceLROnPlateau + early-stopping training loop and the
  combined NSE/MSE loss used in AareML,
* optional *future covariates* fed to the decoder at each forecast step
  (e.g. weather forecasts, day of year),
* optional per-step loss weights, e.g. ``[0.5, 0.3, 0.2]`` to favour day 1.

Requires the optional ``ml`` extra: ``pip install -e .[ml]``.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("seq2seq needs PyTorch: pip install -e '.[ml]'") from e


class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_feat: int, n_tgt: int, horizon: int,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2, n_future: int = 0):
        super().__init__()
        self.n_tgt, self.horizon, self.n_future = n_tgt, horizon, n_future
        drop = dropout if n_layers > 1 else 0.0
        self.encoder = nn.LSTM(n_feat, hidden, n_layers, dropout=drop, batch_first=True)
        self.decoder = nn.LSTM(n_tgt + n_future, hidden, n_layers, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_tgt)

    def forward(self, x, teacher_forcing_ratio: float = 0.0, y_target=None, x_future=None):
        _, (h, c) = self.encoder(x)
        dec_in = torch.zeros(x.size(0), 1, self.n_tgt, device=x.device)
        outs = []
        for t in range(self.horizon):
            step_in = dec_in if x_future is None else torch.cat([dec_in, x_future[:, t:t + 1]], -1)
            out, (h, c) = self.decoder(step_in, (h, c))
            pred = self.fc(self.drop(out))
            outs.append(pred)
            use_tf = (teacher_forcing_ratio > 0 and y_target is not None
                      and torch.rand(1).item() < teacher_forcing_ratio)
            dec_in = y_target[:, t:t + 1, :] if use_tf else pred.detach()
        return torch.cat(outs, dim=1)


def nse_mse_loss(pred, target, alpha: float = 0.5, step_weights=None):
    """AareML loss: alpha * MSE/var + (1 - alpha) * MSE.

    ``step_weights`` (length = horizon, normalised to sum 1) weights the MSE of
    each forecast step; ``None`` = uniform, identical to the AareML loss.
    """
    per_step = torch.mean((pred - target) ** 2, dim=(0, 2))  # [H]
    if step_weights is None:
        mse = per_step.mean()
    else:
        mse = torch.sum(per_step * step_weights)
    var = torch.var(target, unbiased=False).clamp(min=1e-8)
    return alpha * mse / var + (1 - alpha) * mse


class Scaler:
    """Per-column standardisation fitted on training data only."""

    def fit(self, a: np.ndarray):
        flat = a.reshape(-1, a.shape[-1])
        self.mean_ = flat.mean(0)
        self.std_ = flat.std(0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, a):
        return ((a - self.mean_) / self.std_).astype(np.float32)

    def inverse(self, a):
        return (a * self.std_ + self.mean_).astype(np.float32)


class Seq2SeqForecaster:
    """Fit / predict wrapper with scaling, early stopping and a fixed seed."""

    name = "seq2seq_lstm"

    def __init__(self, horizon: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.2,
                 lr: float = 1e-3, epochs: int = 100, patience: int = 12, batch_size: int = 64,
                 teacher_forcing_start: float = 0.5, seed: int = 42,
                 step_weights: list[float] | None = None, verbose: bool = True):
        if step_weights is not None and len(step_weights) != horizon:
            raise ValueError(f"step_weights needs {horizon} values, got {len(step_weights)}")
        self.params = dict(horizon=horizon, hidden=hidden, n_layers=n_layers, dropout=dropout,
                           lr=lr, epochs=epochs, patience=patience, batch_size=batch_size,
                           teacher_forcing_start=teacher_forcing_start, seed=seed,
                           step_weights=step_weights)
        self.verbose = verbose
        self.history: dict[str, list[float]] = {"train": [], "val": []}

    def fit(self, X_tr, y_tr, X_va, y_va, Xf_tr=None, Xf_va=None, xf_tr_augment=None):
        """``xf_tr_augment(epoch) -> Xf_tr``: optional, a fresh (e.g. noise-augmented)
        copy of the training future covariates for each epoch."""
        p = self.params
        torch.manual_seed(p["seed"])
        np.random.seed(p["seed"])
        self.xs, self.ys = Scaler().fit(X_tr), Scaler().fit(y_tr)
        Xt, yt = torch.tensor(self.xs.transform(X_tr)), torch.tensor(self.ys.transform(y_tr))
        Xv, yv = torch.tensor(self.xs.transform(X_va)), torch.tensor(self.ys.transform(y_va))
        self.fs = Scaler().fit(Xf_tr) if Xf_tr is not None else None
        Ft = torch.tensor(self.fs.transform(Xf_tr)) if self.fs else None
        Fv = torch.tensor(self.fs.transform(Xf_va)) if self.fs else None
        sw = p["step_weights"]
        sw = torch.tensor(np.asarray(sw, np.float32) / np.sum(sw)) if sw is not None else None
        self.model = Seq2SeqLSTM(X_tr.shape[-1], y_tr.shape[-1], p["horizon"],
                                 p["hidden"], p["n_layers"], p["dropout"],
                                 n_future=Xf_tr.shape[-1] if Xf_tr is not None else 0)
        opt = torch.optim.AdamW(self.model.parameters(), lr=p["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator().manual_seed(p["seed"])
        for epoch in range(1, p["epochs"] + 1):
            tf = p["teacher_forcing_start"] * max(0.0, 1 - epoch / (p["epochs"] * 0.5))
            if xf_tr_augment is not None and Ft is not None:
                Ft = torch.tensor(self.fs.transform(xf_tr_augment(epoch)))
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            tot = 0.0
            for i in range(0, len(Xt), p["batch_size"]):
                idx = perm[i:i + p["batch_size"]]
                opt.zero_grad()
                fut = Ft[idx] if Ft is not None else None
                loss = nse_mse_loss(self.model(Xt[idx], tf, yt[idx], fut), yt[idx], step_weights=sw)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            self.model.eval()
            with torch.no_grad():
                val = nse_mse_loss(self.model(Xv, x_future=Fv), yv, step_weights=sw).item()
            sched.step(val)
            self.history["train"].append(tot / len(Xt))
            self.history["val"].append(val)
            if val < best:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= p["patience"]:
                    break
            if self.verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:3d}  train={tot / len(Xt):.4f}  val={val:.4f}")
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmin(self.history["val"]) + 1)
        return self

    def predict(self, X, Xf=None) -> np.ndarray:
        self.model.eval()
        fut = torch.tensor(self.fs.transform(Xf)) if self.fs is not None else None
        with torch.no_grad():
            out = self.model(torch.tensor(self.xs.transform(X)), x_future=fut).numpy()
        return self.ys.inverse(out)

    def save(self, path):
        torch.save({"state": self.model.state_dict(), "params": self.params,
                    "x_mean": self.xs.mean_, "x_std": self.xs.std_,
                    "y_mean": self.ys.mean_, "y_std": self.ys.std_,
                    "f_mean": self.fs.mean_ if self.fs else None,
                    "f_std": self.fs.std_ if self.fs else None}, path)
