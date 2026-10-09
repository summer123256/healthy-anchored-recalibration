"""
训练与评估（PyTorch）
====================
示例：
  python train.py --model tiny --dataset cwru                        # 四种负载混合训练
  python train.py --model tiny --dataset cwru --train-loads 0 --test-loads 3   # 跨负载
  python train.py --model tiny --dataset cwru --noise-aug            # 训练时加噪声增强
  python train.py --model tiny --dataset pu                          # 帕德博恩数据，源域 N15_M07_F10
  python train.py --model tiny --dataset fan                         # 风扇台源域 S0（可选）

输出：checkpoints/<名称>.pt  以及 results/<名称>.json（准确率、F1、参数量、MACs 等）
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from config import CKPT_DIR, CWRU_CLASSES, FAN_CLASSES, PU_CLASSES, RESULT_DIR
from models import build_model, count_macs, count_params
from preprocess import windows_to_input


def load_dataset(args):
    if args.dataset == "cwru":
        from data_cwru import build_cwru
        loads = sorted(set(args.train_loads) | set(args.test_loads))
        ds = build_cwru(loads=tuple(loads), channel=args.channel)
        # 跨负载：训练/验证只用 train_loads，测试只用 test_loads
        tr, va, te = ds["train"], ds["val"], ds["test"]
        sel = lambda d, ls: tuple(a[np.isin(d[2], ls)] for a in d)  # noqa: E731
        return (sel(tr, args.train_loads), sel(va, args.train_loads),
                sel(te, args.test_loads), CWRU_CLASSES)
    if args.dataset == "pu":
        from data_pu import build_pu_source
        ds = build_pu_source()
        return ds["train"], ds["val"], ds["test"], PU_CLASSES
    from data_fan import build_fan_source
    ds = build_fan_source()
    return ds["train"], ds["val"], ds["test"], FAN_CLASSES


def to_tensor(X, snr=None, rng=None):
    xin, _ = windows_to_input(X, snr_db=snr, rng=rng)
    return torch.from_numpy(xin)


@torch.no_grad()
def predict(model, X, batch=512, snr=None, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(0, len(X), batch):
        preds.append(model(to_tensor(X[i:i + batch], snr, rng)).argmax(1).numpy())
    return np.concatenate(preds)


def train_model(model, train, val, epochs=60, lr=1e-3, batch=64, noise_aug=False, seed=0,
                verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    Xva_t = to_tensor(Xva)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(ytr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            snr = float(rng.uniform(-4, 10)) if (noise_aug and rng.random() < 0.5) else None
            xb = to_tensor(Xtr[idx], snr, rng)
            yb = torch.from_numpy(ytr[idx])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(Xva_t).argmax(1).numpy() == yva).mean()
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}/{epochs}  loss {tot / len(ytr):.4f}  val_acc {va_acc:.4f}")
    model.load_state_dict(best_state)
    return best_acc


def evaluate(model, test, n_classes, snr=None, seed=0):
    X, y, _ = test
    p = predict(model, X, snr=snr, seed=seed)
    return dict(acc=float((p == y).mean()),
                macro_f1=float(f1_score(y, p, average="macro")),
                cm=confusion_matrix(y, p, labels=list(range(n_classes))).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--dataset", default="cwru", choices=["cwru", "pu", "fan"])
    ap.add_argument("--train-loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--test-loads", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--channel", default="DE", choices=["DE", "FE"], help="CWRU 使用哪个加速度计")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--noise-aug", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    train, val, test, classes = load_dataset(args)
    print(f"训练 {len(train[1])}，验证 {len(val[1])}，测试 {len(test[1])} 个样本")
    model = build_model(args.model, len(classes))
    t0 = time.time()
    val_acc = train_model(model, train, val, args.epochs, args.lr, args.batch, args.noise_aug, args.seed)
    res = evaluate(model, test, len(classes))
    res.update(model=args.model, dataset=args.dataset, val_acc=float(val_acc),
               params=count_params(model), macs=count_macs(model), seed=args.seed,
               train_loads=args.train_loads, test_loads=args.test_loads,
               noise_aug=args.noise_aug, train_seconds=round(time.time() - t0, 1))
    name = args.name or f"{args.dataset}_{args.model}_s{args.seed}{'_na' if args.noise_aug else ''}"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), model=args.model, n_classes=len(classes),
                    classes=classes, args=vars(args)), CKPT_DIR / f"{name}.pt")
    (RESULT_DIR / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"测试准确率 {res['acc']:.4f}，Macro-F1 {res['macro_f1']:.4f}，"
          f"参数量 {res['params']}，MACs {res['macs']}")
    print(f"已保存 checkpoints/{name}.pt")


if __name__ == "__main__":
    main()
