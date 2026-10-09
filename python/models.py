"""
模型定义（PyTorch）
==================
所有模型都写成 nn.Sequential，只用下列层，保证能导出到 STM32：
  Conv1d / Linear / BatchNorm1d / ReLU / MaxPool1d / GAP / Flatten / Dropout

  tiny        本文方法：宽卷积首层 + 深度可分离卷积 + 全局平均池化
  tiny_std    消融：深度可分离卷积换成普通卷积
  tiny_k8..   消融：首层卷积核长度 8/16/32/128
  wdcnn       对比：WDCNN (Zhang et al., Sensors 2017)，按 1024 点输入改写
  plaincnn    对比：常规 1D-CNN（逐层小卷积 + 全连接）
"""
import torch
import torch.nn as nn


class GAP(nn.Module):
    """全局平均池化：(N, C, L) -> (N, C)"""

    def forward(self, x):
        return x.mean(dim=-1)


def _cbr(cin, cout, k, stride=1, pad=0, groups=1):
    return [nn.Conv1d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]


def tiny(n_classes=10, k1=64, separable=True, widths=(16, 32, 32, 64)):
    """本文提出的超轻量网络。输入 (N,1,1024)。
    首层：k1 长卷积，步长 8，输出长度固定 128（要求 k1 为 >=8 的偶数）。"""
    assert k1 >= 8 and k1 % 2 == 0
    c0, c1, c2, c3 = widths
    layers = _cbr(1, c0, k1, stride=8, pad=(k1 - 8) // 2)
    layers += [nn.MaxPool1d(2)]

    def block(cin, cout):
        if separable:
            return _cbr(cin, cin, 3, pad=1, groups=cin) + _cbr(cin, cout, 1)
        return _cbr(cin, cout, 3, pad=1)

    layers += block(c0, c1) + [nn.MaxPool1d(2)]
    layers += block(c1, c2) + [nn.MaxPool1d(2)]
    layers += block(c2, c3)
    layers += [GAP(), nn.Linear(c3, n_classes)]
    return nn.Sequential(*layers)


def wdcnn(n_classes=10):
    """WDCNN（Zhang W. et al., Sensors 2017, 17(2):425）。原文输入 2048 点，
    这里输入 1024 点，其余结构保持一致：首层 16@64/16，后接 4 个 3 点卷积层。"""
    layers = _cbr(1, 16, 64, stride=16, pad=24) + [nn.MaxPool1d(2)]   # 64 -> 32
    layers += _cbr(16, 32, 3, pad=1) + [nn.MaxPool1d(2)]              # 16
    layers += _cbr(32, 64, 3, pad=1) + [nn.MaxPool1d(2)]              # 8
    layers += _cbr(64, 64, 3, pad=1) + [nn.MaxPool1d(2)]              # 4
    layers += _cbr(64, 64, 3, pad=0) + [nn.MaxPool1d(2)]              # 2 -> 1
    layers += [nn.Flatten(), nn.Linear(64, 100), nn.BatchNorm1d(100), nn.ReLU(inplace=True),
               nn.Linear(100, n_classes)]
    return nn.Sequential(*layers)


def plaincnn(n_classes=10):
    """常规 1D-CNN 基线：不做轻量化设计。"""
    layers = _cbr(1, 16, 15, pad=7) + [nn.MaxPool1d(4)]     # 256
    layers += _cbr(16, 32, 7, pad=3) + [nn.MaxPool1d(4)]    # 64
    layers += _cbr(32, 64, 5, pad=2) + [nn.MaxPool1d(4)]    # 16
    layers += _cbr(64, 64, 3, pad=1) + [nn.MaxPool1d(4)]    # 4
    layers += [nn.Flatten(), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
               nn.Linear(64, n_classes)]
    return nn.Sequential(*layers)


MODEL_ZOO = {
    "tiny": lambda n: tiny(n),
    "tiny_std": lambda n: tiny(n, separable=False),
    "tiny_k8": lambda n: tiny(n, k1=8),
    "tiny_k16": lambda n: tiny(n, k1=16),
    "tiny_k32": lambda n: tiny(n, k1=32),
    "tiny_k128": lambda n: tiny(n, k1=128),
    "wdcnn": wdcnn,
    "plaincnn": plaincnn,
}


def build_model(name, n_classes):
    return MODEL_ZOO[name](n_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, input_len=1024):
    """统计一次推理的乘加次数（MACs），只统计 Conv1d 与 Linear。"""
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0]

    def fc_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, input_len))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return total[0]


# ---------------------------------------------------------------------------
# 导出为与框架无关的层列表（numpy），供 quant.py 量化和生成 C 代码
# ---------------------------------------------------------------------------
def _np(t):
    return None if t is None else t.detach().cpu().double().numpy()


def _bn(m):
    return dict(gamma=_np(m.weight), beta=_np(m.bias), mean=_np(m.running_mean),
                var=_np(m.running_var), eps=float(m.eps))


def export_graph(model):
    mods = [m for m in model.modules() if not isinstance(m, nn.Sequential)]
    graph, i = [], 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            layer = dict(type="conv" if isinstance(m, nn.Conv1d) else "fc",
                         w=_np(m.weight), b=_np(m.bias), bn=None, relu=False)
            if isinstance(m, nn.Conv1d):
                layer.update(stride=m.stride[0], pad=m.padding[0], groups=m.groups)
            j = i + 1
            if j < len(mods) and isinstance(mods[j], nn.BatchNorm1d):
                layer["bn"] = _bn(mods[j])
                j += 1
            if j < len(mods) and isinstance(mods[j], nn.ReLU):
                layer["relu"] = True
                j += 1
            graph.append(layer)
            i = j
            continue
        if isinstance(m, nn.MaxPool1d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            assert k == s and m.padding == 0, "只支持 kernel==stride 且无填充的最大池化"
            graph.append(dict(type="maxpool", k=k))
        elif isinstance(m, GAP):
            graph.append(dict(type="gap"))
        elif isinstance(m, nn.Flatten):
            graph.append(dict(type="flatten"))
        elif isinstance(m, nn.Dropout):
            pass
        else:
            raise TypeError(f"不支持导出的层：{type(m).__name__}")
        i += 1
    assert graph[-1]["type"] == "fc", "最后一层必须是 Linear"
    return graph
