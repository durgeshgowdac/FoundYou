import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from config import Config

from logger import log

# ======================== OSNet Backbone =========================
class _ConvBnRelu(nn.Module):
    def __init__(self, ic, oc, k, s=1, p=0, g=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ic, oc, k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
    def forward(self, x): return self.f(x)

class _LiteConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.f = nn.Sequential(_ConvBnRelu(ic, ic, 3, p=1, g=ic),
                                _ConvBnRelu(ic, oc, 1))
    def forward(self, x): return self.f(x)

class _OSBlock(nn.Module):
    def __init__(self, ic, oc, r=4):
        super().__init__()
        mid = max(oc // r, 16)
        self.c1 = _ConvBnRelu(ic, mid, 1)
        self.streams = nn.ModuleList([
            _LiteConv(mid, mid),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid)),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid), _LiteConv(mid, mid)),
            nn.Sequential(_LiteConv(mid, mid), _LiteConv(mid, mid),
                          _LiteConv(mid, mid), _LiteConv(mid, mid)),
        ])
        gm = max(mid // 4, 1)
        self.gate = nn.Sequential(nn.Linear(mid, gm), nn.ReLU(inplace=True),
                                   nn.Linear(gm, 4), nn.Sigmoid())
        self.c2   = _ConvBnRelu(mid, oc, 1)
        self.skip = nn.Sequential(_ConvBnRelu(ic, oc, 1)) if ic != oc else nn.Identity()

    def forward(self, x):
        br = [s(self.c1(x)) for s in self.streams]
        w  = self.gate(sum(b.mean([2,3]) for b in br)/4).unsqueeze(-1).unsqueeze(-1)
        return F.relu(self.skip(x) + self.c2(sum(w[:,i]*br[i] for i in range(4))),
                      inplace=True)

class OSNet(nn.Module):
    CH = [64, 256, 384, 512]
    def __init__(self, fdim=512):
        super().__init__()
        c = self.CH
        self.stem   = nn.Sequential(_ConvBnRelu(3, c[0], 7, s=2, p=3),
                                     nn.MaxPool2d(3, stride=2, padding=1))
        self.layer2 = self._layer(c[0], c[1], 2, True)
        self.layer3 = self._layer(c[1], c[2], 2, True)
        self.layer4 = self._layer(c[2], c[3], 2, False)
        self.head   = nn.Sequential(_ConvBnRelu(c[3], c[3], 1),
                                     nn.AdaptiveAvgPool2d(1))
        self.fc     = nn.Linear(c[3], fdim)

    @staticmethod
    def _layer(ic, oc, n, ds):
        layers = [_OSBlock(ic, oc)] + [_OSBlock(oc, oc) for _ in range(n-1)]
        if ds:
            layers += [nn.Sequential(nn.Conv2d(oc,oc,2,stride=2,bias=False),
                                      nn.BatchNorm2d(oc), nn.ReLU(inplace=True))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.head(x).flatten(1))


def _build_backbone():
    """Returns (net, feature_dim). Does NOT mutate Config."""
    try:
        import torchreid
        src = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=1, pretrained=True)
        src.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 128)
            out   = src(dummy)
        fdim = out.shape[1]
        log.info(f"OSNet via torchreid - fdim={fdim}")
        return src, fdim
    except Exception as e:
        log.warning(f"torchreid not usable ({e}), falling back to custom OSNet")

    fdim  = Config.FEATURE_DIM
    model = OSNet(fdim)
    cache = os.path.expanduser("~/.cache/osnet/osnet_x1_0_imagenet.pth")
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if not os.path.exists(cache):
        try:
            import gdown
            log.info("Downloading OSNet weights...")
            gdown.download(
                "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
                cache, quiet=False)
        except Exception as e:
            log.warning(f"Download failed: {e} - random weights (poor ReID)")
            return model, fdim

    try:
        raw = torch.load(cache, map_location='cpu')
        sd  = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
        dst = model.state_dict()
        ok  = {k: v for k, v in sd.items()
               if k in dst and dst[k].shape == v.shape}
        if len(ok) == 0:
            log.warning("0 keys matched - random weights. Install torchreid.")
        else:
            model.load_state_dict(ok, strict=False)
            log.info(f"OSNet from cache ({len(ok)}/{len(dst)} keys)")
    except Exception as e:
        log.warning(f"Cache load failed: {e} - random weights")

    return model, fdim


# ======================== FEATURE EXTRACTOR =========================
class FeatureExtractor:
    def __init__(self):
        log.info(f"Building OSNet on {Config.DEVICE}...")
        net, fdim = _build_backbone()
        if fdim != Config.FEATURE_DIM:
            log.info(f"Updating FEATURE_DIM: {Config.FEATURE_DIM} -> {fdim}")
            Config.FEATURE_DIM = fdim
        self.fdim = fdim
        self.net  = net.to(Config.DEVICE).eval()
        self.tf   = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        log.info(f"FeatureExtractor ready (dim={self.fdim})")

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(batch), p=2, dim=1)

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        out     = [None] * len(crops)
        tensors = []
        valid   = []
        for i, c in enumerate(crops):
            if c is None or c.size == 0 or c.shape[0] <= 10 or c.shape[1] <= 5:
                continue
            try:
                t = self.tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).unsqueeze(0)
                tensors.append(t); valid.append(i)
            except Exception:
                continue

        if not tensors:
            return out

        try:
            B  = torch.cat(tensors).to(Config.DEVICE)
            f  = self._forward(B)
            ff = self._forward(torch.flip(B, [3]))
            fs = F.normalize((f + ff) / 2, p=2, dim=1).cpu().numpy()
            for ai, ci in enumerate(valid):
                v = fs[ai].astype(np.float32)
                n = np.linalg.norm(v)
                if n > 1e-6 and not np.isnan(v).any():
                    out[ci] = v / n
        except Exception as e:
            if Config.DEBUG:
                log.debug(f"Embed batch failed: {e}, per-crop fallback")
            for ai, ci in enumerate(valid):
                try:
                    t = tensors[ai].to(Config.DEVICE)
                    v = self._forward(t).cpu().numpy()[0]
                    n = np.linalg.norm(v)
                    if n > 1e-6 and not np.isnan(v).any():
                        out[ci] = (v/n).astype(np.float32)
                except Exception:
                    pass
        return out