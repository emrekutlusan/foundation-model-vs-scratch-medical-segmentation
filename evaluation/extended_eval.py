"""Extended evaluation of the saved checkpoints: six segmentation metrics per image.

No model is trained or selected here. Every checkpoint is re-run on the test sets at
the resolution it was trained and evaluated at, the per-image Dice is first checked
against the values reported in the paper, and only then are the other metrics written.

Usage, in each domain notebook (polyp.ipynb, breast.ipynb, skin.ipynb), after the cells
that build `clean_sets` have run:

    import sys; sys.path.insert(0, '/content/drive/MyDrive/thesis_ckpts')
    import extended_eval as ee
    ee.run_domain('polyp', clean_sets, CKPT)      # or 'breast' / 'skin'

Outputs, in CKPT/extended_eval/:
    metrics_<domain>.csv          one row per (checkpoint, test set, image)
    validation_<domain>.txt       recomputed vs reported Dice, with the largest deviation
    qualitative_<domain>.npz      predictions of the three models of the main comparison
"""
import os, io, math, time, datetime
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage

PATCH = 14
# NSD tolerance as a fraction of the image width: 2 px at 224 (primary) and 4 px at 224
# (secondary). The same fractions are applied at every resolution.
TOL_FRACTIONS = (2 / 224, 4 / 224)
DICE_TOL = 0.001          # allowed |recomputed - reported| for set-mean Dice, plus rounding
QUAL_SIZE = 256           # display size of images stored for the qualitative figure
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------------- data
class EvalDataset(Dataset):
    """Resize + ImageNet normalisation, identical to the eval-mode datasets of the
    notebooks (PolypDataset, SegDataset, ISICDataset, SegDatasetStrong, SegDataset224
    all reduce to this when augment=False; only the size differs)."""
    def __init__(self, imgs, masks, size):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        self.imgs, self.masks = list(imgs), list(masks)
        self.t = A.Compose([A.Resize(size, size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        image = np.array(Image.open(self.imgs[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks[i]).convert('L'))
        t = self.t(image=image, mask=mask)
        return t['image'].float(), (t['mask'].float() / 255.0 > 0.5).float().unsqueeze(0), i


# ----------------------------------------------------------------------------- models
def _dinov2_small():
    import transformers
    from transformers import AutoModel
    # The checkpoints were saved with transformers 4.x, where DINOv2 attention modules are
    # named query/key/value. From 5.0 they are q_proj/k_proj/v_proj, so neither the LoRA
    # targets nor the saved keys would match.
    if int(transformers.__version__.split('.')[0]) >= 5:
        raise ImportError(
            f'transformers {transformers.__version__} renames the DINOv2 modules used by the '
            'checkpoints. Run:  !pip install -q "transformers==4.57.6"  then restart the '
            'runtime and rerun the setup cells.')
    return AutoModel.from_pretrained('facebook/dinov2-small')

ENCODER_FACTORY = _dinov2_small        # replaced only in offline tests


class LinearDecoder(nn.Module):
    def __init__(self, size, embed_dim=384):
        super().__init__()
        self.size, self.grid = size, size // PATCH
        self.proj = nn.Conv2d(embed_dim, 1, 1)
    def forward(self, tokens):
        B, N, D = tokens.shape
        x = tokens.permute(0, 2, 1).reshape(B, D, self.grid, self.grid)
        return F.interpolate(self.proj(x), size=self.size, mode='bilinear', align_corners=False)


class DinoSeg(nn.Module):
    """Frozen, BitFit, LoRA and full fine-tuning share this structure; their checkpoints
    differ only in which tensors were trained, so one class loads all of them."""
    def __init__(self, size, lora_rank=None):
        super().__init__()
        enc = ENCODER_FACTORY()
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                             target_modules=['query', 'value'], lora_dropout=0.1, bias='none')
            enc = get_peft_model(enc, cfg)
        self.encoder, self.decoder = enc, LinearDecoder(size)
    def forward(self, x):
        return self.decoder(self.encoder(pixel_values=x).last_hidden_state[:, 1:, :])


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.InstanceNorm2d(cout, affine=True)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.InstanceNorm2d(cout, affine=True))
                     if (stride != 1 or cin != cout) else nn.Identity())
    def forward(self, x):
        r = self.skip(x)
        x = self.act(self.n1(self.c1(x)))
        return self.act(self.n2(self.c2(x)) + r)


class PlainBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.InstanceNorm2d(cout, affine=True)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.InstanceNorm2d(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
    def forward(self, x):
        x = self.act(self.n1(self.c1(x)))
        return self.act(self.n2(self.c2(x)))


class UNet(nn.Module):
    """ResUNet / PlainUNet of the notebooks; in eval mode the forward returns the
    full-resolution logits only (deep-supervision heads are unused)."""
    def __init__(self, size, block, base=32, depth=5):
        super().__init__()
        self.size, self.deep_sup = size, True
        ch = [base * 2 ** i for i in range(depth)]
        self.stem = block(3, ch[0])
        self.enc = nn.ModuleList([block(ch[i], ch[i + 1], stride=2) for i in range(depth - 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2)
                                 for i in reversed(range(depth - 1))])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(depth - 1))])
        self.head = nn.ModuleList([nn.Conv2d(ch[i], 1, 1) for i in reversed(range(depth - 1))])
    def forward(self, x):
        feats = [self.stem(x)]
        for e in self.enc: feats.append(e(feats[-1]))
        d, outs = feats[-1], []
        for i, (u, dec, hd) in enumerate(zip(self.up, self.dec, self.head)):
            d = dec(torch.cat([u(d), feats[-2 - i]], dim=1))
            outs.append(hd(d))
        return F.interpolate(outs[-1], size=self.size, mode='bilinear', align_corners=False)


def build_model(run):
    a, s = run['arch'], run['size']
    if a == 'dino':    return DinoSeg(s, lora_rank=run.get('rank'))
    if a == 'resunet': return UNet(s, ResBlock)
    if a == 'plain':   return UNet(s, PlainBlock)
    raise ValueError(a)


def load_checkpoint(run, ckpt_dir):
    m = build_model(run)
    sd = torch.load(os.path.join(ckpt_dir, run['tag'] + '.pth'), map_location='cpu')
    res = m.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"{run['tag']}: checkpoint does not match the model definition.\n"
            f"  missing ({len(res.missing_keys)}): {res.missing_keys[:5]}\n"
            f"  unexpected ({len(res.unexpected_keys)}): {res.unexpected_keys[:5]}\n"
            "  If the key names differ only in the encoder, the installed transformers/peft "
            "version names DINOv2 modules differently from the one used for training; "
            "install the training-time versions and rerun.")
    return m.to(DEVICE).eval()


# ----------------------------------------------------------------------------- metrics
_CROSS = ndimage.generate_binary_structure(2, 1)

def _border(m):
    return m & ~ndimage.binary_erosion(m, structure=_CROSS, border_value=0)

def surface_metrics(pred, gt, size):
    """HD95, ASSD and NSD at each tolerance for one binary pair, in pixels.

    Both empty: perfect agreement (0 distance, NSD 1).
    Exactly one empty: the maximum possible distance, the image diagonal of this
    resolution, and NSD 0. Images where this happens are flagged in the output."""
    diag = size * math.sqrt(2.0)
    taus = [f * size for f in TOL_FRACTIONS]
    p_any, g_any = bool(pred.any()), bool(gt.any())
    if not p_any and not g_any:
        return 0.0, 0.0, [1.0] * len(taus)
    if not p_any or not g_any:
        return diag, diag, [0.0] * len(taus)
    bp, bg = _border(pred), _border(gt)
    d_pg = ndimage.distance_transform_edt(~bg)[bp]     # pred surface -> gt surface
    d_gp = ndimage.distance_transform_edt(~bp)[bg]     # gt surface -> pred surface
    both = np.concatenate([d_pg, d_gp])
    hd95 = float(np.percentile(both, 95))
    assd = float((d_pg.mean() + d_gp.mean()) / 2.0)
    nsd = [float(((d_pg <= t).sum() + (d_gp <= t).sum()) / both.size) for t in taus]
    return hd95, assd, nsd


def overlap_metrics(p, y):
    """Dice and IoU exactly as in the notebooks (same epsilon), plus sensitivity and
    precision. p, y: float tensors (B,1,H,W) of 0/1."""
    inter = (p * y).sum(dim=(1, 2, 3))
    psum, ysum = p.sum(dim=(1, 2, 3)), y.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (psum + ysum + 1e-6)
    iou = (inter + 1e-6) / (psum + ysum - inter + 1e-6)
    return dice, iou, inter, psum, ysum


@torch.no_grad()
def evaluate(model, imgs, masks, size, keep_preds=False, batch_size=8):
    ds = EvalDataset(imgs, masks, size)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    rows, preds = [], []
    for x, y, idx in ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == 'cuda'):
            lo = model(x)
        p = (torch.sigmoid(lo.float()) > 0.5).float()
        dice, iou, inter, psum, ysum = overlap_metrics(p, y)
        pn, yn = p[:, 0].bool().cpu().numpy(), y[:, 0].bool().cpu().numpy()
        for k in range(x.shape[0]):
            hd95, assd, nsd = surface_metrics(pn[k], yn[k], size)
            I, P, G = float(inter[k]), float(psum[k]), float(ysum[k])
            diag = size * math.sqrt(2.0)
            rows.append(dict(
                image=os.path.basename(str(ds.imgs[int(idx[k])])),
                dice=float(dice[k]), iou=float(iou[k]),
                sensitivity=I / G if G > 0 else np.nan,
                # empty prediction with a lesion present counts as precision 0 (a failure);
                # undefined only when both are empty
                precision=I / P if P > 0 else (0.0 if G > 0 else np.nan),
                hd95_px=hd95, assd_px=assd,
                hd95_norm=hd95 / diag, assd_norm=assd / diag,
                **{f'nsd_{j + 1}': v for j, v in enumerate(nsd)},
                pred_empty=int(P == 0), gt_empty=int(G == 0),
                gt_frac=G / (size * size), pred_frac=P / (size * size)))
            if keep_preds: preds.append(np.packbits(pn[k]))
    return rows, preds


# ----------------------------------------------------------------------------- runs
def _r(rid, group, arch, tag, size, rank=None, seed=42):
    return dict(id=rid, group=group, arch=arch, tag=tag, size=size, rank=rank, seed=seed)

SEEDS = (42, 123, 2024)

def runs_for(domain):
    R = []
    if domain == 'polyp':
        S = 224
        R += [_r('plain', 'main', 'plain', 'plainunet_polyp_lr0.001_e300_s42', S),
              _r('resunet', 'main', 'resunet', 'resunet_polyp_lr0.003_e300_s42', S),
              _r('lora32_e40', 'main', 'dino', 'lora_r32_polyp_s42_e40', S, 32)]
        R += [_r('frozen_e40', 'capacity', 'dino', 'frozen_polyp_s42_e40', S),
              _r('bitfit_e40', 'capacity', 'dino', 'bitfit_polyp_lr1e-03_s42', S)]
        R += [_r(f'lora{r}_e40', 'capacity', 'dino', f'lora_r{r}_polyp_s42_e40', S, r) for r in (4, 8, 16)]
        R += [_r('full_e40', 'capacity', 'dino', 'full_polyp_s42_e40', S)]
        R += [_r(f'lora32_s{s}', 'seeds', 'dino', f'lora_r32_polyp_s{s}', S, 32, s) for s in SEEDS]
        R += [_r(f'full_s{s}', 'seeds', 'dino', f'full_polyp_s{s}', S, None, s) for s in SEEDS]
        R += [_r('lora4_e20', 'budget', 'dino', 'lora_r4_polyp_s42', S, 4)]
        for f in (10, 25, 50):
            R += [_r(f'lora32_f{f}', 'datasize', 'dino', f'lora_r32_polyp_f{f}_s42', S, 32),
                  _r(f'resunet_f{f}', 'datasize', 'resunet', f'resunet_polyp_f{f}_s42', S)]
    elif domain == 'breast':
        S = 224
        R += [_r('plain', 'main', 'plain', 'plainunet_breast_lr0.001_e300_s42', S),
              _r('resunet', 'main', 'resunet', 'resunet_breast_lr0.003_e300_s42', S),
              _r('lora32_e40', 'main', 'dino', 'lora_r32_breast_s42_e40', S, 32)]
        R += [_r('frozen_e40', 'capacity', 'dino', 'frozen_breast_s42_e40', S),
              _r('bitfit_e40', 'capacity', 'dino', 'bitfit_breast_lr3e-03_s42', S)]
        R += [_r(f'lora{r}_e40', 'capacity', 'dino', f'lora_r{r}_breast_s42_e40', S, r) for r in (4, 8, 16)]
        R += [_r('full_e40', 'capacity', 'dino', 'full_breast_s42_e40', S)]
        R += [_r(f'lora{r}_e20', 'budget', 'dino', f'lora_r{r}_breast_s42', S, r) for r in (4, 16)]
        for f in (25, 50):
            R += [_r(f'lora32_f{f}', 'datasize', 'dino', f'lora_r32_breast_f{f}_s42', S, 32),
                  _r(f'resunet_f{f}', 'datasize', 'resunet', f'resunet_breast_f{f}_s42', S)]
    elif domain == 'skin':
        # DINOv2 at 392 (multiple of the 14-px patch), U-Nets at 384 (divisible by 32),
        # the data-size sweep at 224 for every model and fraction.
        R += [_r('plain', 'main', 'plain', 'plainunet_skin_lr0.003_e100_b16_s42', 384),
              _r('resunet', 'main', 'resunet', 'resunet_skin_lr0.003_e100_b16_s42', 384)]
        R += [_r(f'lora32_s{s}', 'main', 'dino', f'lora_r32_s{s}', 392, 32, s) for s in SEEDS]
        R += [_r('bitfit', 'capacity', 'dino', 'bitfit_skin_lr1e-03_s42', 392)]
        R += [_r(f'lora{r}', 'capacity', 'dino', f'lora_r{r}_s42', 392, r) for r in (4, 8, 16)]
        R += [_r(f'lora8ms_s{s}', 'seeds', 'dino', f'lora_s{s}', 392, 8, s) for s in SEEDS]
        R += [_r(f'full_s{s}', 'seeds', 'dino', f'full_s{s}', 392, None, s) for s in SEEDS]
        for f in (25, 50, 100):
            R += [_r(f'lora32_224_f{f}', 'datasize', 'dino', f'lora_r32_skin224_f{f}_s42', 224, 32),
                  _r(f'resunet_224_f{f}', 'datasize', 'resunet', f'resunet_skin224_f{f}_v2_s42', 224)]
    else:
        raise ValueError(domain)
    return R


SETS = {
    'polyp': dict(ID=['Kvasir', 'CVC-ClinicDB'], OOD=['CVC-ColonDB', 'ETIS', 'CVC-T'],
                  n={'Kvasir': 100, 'CVC-ClinicDB': 62, 'CVC-ColonDB': 380, 'ETIS': 196, 'CVC-T': 60}),
    'breast': dict(ID=['BUSI-val'], OOD=['BrEaST', 'BUS-UCLM'],
                   n={'BUSI-val': 54, 'BrEaST': 252, 'BUS-UCLM': 260}),
    'skin': dict(ID=[], OOD=['ISIC 2016*', 'ISIC 2017*', 'PH2'],
                 n={'ISIC 2016*': 269, 'ISIC 2017*': 172, 'PH2': 200}),
}


# ----------------------------------------------------------------------------- checks
# Each check: (label, group, function(mean) -> value, reported value, decimals reported).
# mean(run_id, set) returns the recomputed set-mean Dice. Reported values are copied
# from the paper's tables; half a unit in the
# last reported decimal is added to DICE_TOL as a rounding allowance.
def _per_set(group, run_id, values, dec=4):
    return [(f'{run_id} | {s}', group, (lambda M, r=run_id, s=s: M(r, s)), v, dec) for s, v in values.items()]

def _seedmean(prefix, s):
    return lambda M: np.mean([M(f'{prefix}_s{x}', s) for x in SEEDS])

def checks_for(domain):
    C = []
    if domain == 'polyp':
        T1 = {'plain': (0.8804, 0.8803, 0.7382, 0.6321, 0.8406),
              'resunet': (0.8611, 0.8826, 0.7240, 0.6179, 0.7941),
              'lora32_e40': (0.8877, 0.8684, 0.7694, 0.6984, 0.8665)}
        order = ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS', 'CVC-T']
        for r, v in T1.items():
            C += _per_set('main', r, dict(zip(order, v)))
        T4 = {'frozen_e40': (0.5875, 0.4947, 0.7612, 0.8005), 'bitfit_e40': (0.7204, 0.6867, 0.8634, 0.8960),
              'lora4_e40': (0.7139, 0.6843, 0.8675, 0.8764), 'lora8_e40': (0.7132, 0.6984, 0.8520, 0.8828),
              'lora16_e40': (0.7346, 0.6711, 0.8749, 0.8851), 'full_e40': (0.7396, 0.7041, 0.8739, 0.8790)}
        for r, v in T4.items():
            C += _per_set('capacity', r, dict(zip(['CVC-ColonDB', 'ETIS', 'CVC-T', 'Kvasir'], v)))
        S8 = {'lora32': (0.8825, 0.8485, 0.7401, 0.6986, 0.8671), 'full': (0.8767, 0.8384, 0.7380, 0.6856, 0.8665)}
        for p, v in S8.items():
            for s, val in zip(order, v):
                C.append((f'{p} 3-seed mean | {s}', 'seeds', _seedmean(p, s), val, 4))
        C.append(('rank effect r32-r4, 20 ep | CVC-ColonDB', 'budget',
                  lambda M: M('lora32_s42', 'CVC-ColonDB') - M('lora4_e20', 'CVC-ColonDB'), 0.046, 3))
        C.append(('rank effect r32-r4, 40 ep | CVC-ColonDB', 'capacity',
                  lambda M: M('lora32_e40', 'CVC-ColonDB') - M('lora4_e40', 'CVC-ColonDB'), 0.055, 3))
        gaps = {10: (-0.101, -0.191), 25: (-0.050, -0.135), 50: (-0.042, -0.132)}
    elif domain == 'breast':
        T1 = {'plain': (0.8177, 0.6545, 0.6320), 'resunet': (0.8244, 0.6463, 0.5981),
              'lora32_e40': (0.8029, 0.7133, 0.7046)}
        order = ['BUSI-val', 'BrEaST', 'BUS-UCLM']
        for r, v in T1.items():
            C += _per_set('main', r, dict(zip(order, v)))
        T4 = {'frozen_e40': (0.6099, 0.6660, 0.6493), 'bitfit_e40': (0.7337, 0.7449, 0.7853),
              'lora4_e40': (0.7165, 0.7087, 0.7572), 'lora8_e40': (0.6957, 0.6979, 0.7751),
              'lora16_e40': (0.7179, 0.7226, 0.7950), 'full_e40': (0.6941, 0.7130, 0.7796)}
        for r, v in T4.items():
            C += _per_set('capacity', r, dict(zip(['BrEaST', 'BUS-UCLM', 'BUSI-val'], v)))
        for s, v20, v40 in (('BrEaST', 0.030, 0.001), ('BUS-UCLM', 0.023, 0.014), ('BUSI-val', 0.040, 0.038)):
            C.append((f'rank effect r16-r4, 20 ep | {s}', 'budget',
                      lambda M, s=s: M('lora16_e20', s) - M('lora4_e20', s), v20, 3))
            C.append((f'rank effect r16-r4, 40 ep | {s}', 'capacity',
                      lambda M, s=s: M('lora16_e40', s) - M('lora4_e40', s), v40, 3))
        gaps = {25: (-0.004, -0.104), 50: (0.003, -0.070)}
    else:
        C += _per_set('main', 'plain', {'ISIC 2016*': 0.9123, 'ISIC 2017*': 0.8336, 'PH2': 0.9167})
        C += _per_set('main', 'resunet', {'ISIC 2016*': 0.9084, 'ISIC 2017*': 0.8356, 'PH2': 0.9032})
        for s, v in (('ISIC 2016*', 0.924), ('ISIC 2017*', 0.872), ('PH2', 0.925)):
            C.append((f'lora32 3-seed mean | {s}', 'main', _seedmean('lora32', s), v, 3))
        T4 = {'bitfit': (0.9238, 0.8684, 0.9208), 'lora4': (0.9236, 0.8710, 0.9205),
              'lora8': (0.9242, 0.8635, 0.9289), 'lora16': (0.9233, 0.8629, 0.9236),
              'lora32_s42': (0.9246, 0.8763, 0.9246), 'full_s42': (0.9215, 0.8717, 0.9239)}
        for r, v in T4.items():
            grp = 'seeds' if r == 'full_s42' else 'capacity'
            C += _per_set(grp, r, dict(zip(['ISIC 2016*', 'ISIC 2017*', 'PH2'], v)))
        for s, d8, d32 in (('ISIC 2016*', -0.004, 0.003), ('ISIC 2017*', -0.017, -0.002), ('PH2', 0.002, 0.003)):
            C.append((f'LoRA r8 - full, 3-seed | {s}', 'seeds',
                      lambda M, s=s: _seedmean('lora8ms', s)(M) - _seedmean('full', s)(M), d8, 3))
            C.append((f'LoRA r32 - full, 3-seed | {s}', 'seeds',
                      lambda M, s=s: _seedmean('lora32', s)(M) - _seedmean('full', s)(M), d32, 3))
        ood = SETS['skin']['OOD']
        for f, g in ((25, -0.013), (50, 0.001), (100, -0.005)):
            C.append((f'data-size OOD gap {f}% (224 px)', 'datasize',
                      lambda M, f=f: np.mean([M(f'resunet_224_f{f}', s) - M(f'lora32_224_f{f}', s) for s in ood]), g, 3))
        return C
    sets = SETS[domain]
    for f, (gid, good) in gaps.items():
        C.append((f'data-size in-domain gap {f}%', 'datasize',
                  lambda M, f=f: np.mean([M(f'resunet_f{f}', s) - M(f'lora32_f{f}', s) for s in sets['ID']]), gid, 3))
        C.append((f'data-size OOD gap {f}%', 'datasize',
                  lambda M, f=f: np.mean([M(f'resunet_f{f}', s) - M(f'lora32_f{f}', s) for s in sets['OOD']]), good, 3))
    return C


def run_checks(checks, groups_done, mean, log):
    """Evaluate the checks whose group is complete. Returns the largest deviation and
    the list of failures."""
    worst, fails = 0.0, []
    for label, group, fn, reported, dec in checks:
        if group not in groups_done: continue
        got = float(fn(mean))
        tol = DICE_TOL + 0.5 * 10 ** (-dec)      # 0.001 plus the rounding of the reported value
        dev = abs(got - reported)
        worst = max(worst, dev)
        ok = dev <= tol
        log(f"  {'ok  ' if ok else 'FAIL'} {label:48s} reported {reported:+.4f}  recomputed {got:+.4f}"
            f"  |dev| {dev:.4f}  (tol {tol:.4f})")
        if not ok: fails.append(label)
    return worst, fails


# ----------------------------------------------------------------------------- driver
def _check_sets(domain, clean_sets):
    exp = SETS[domain]['n']
    for s, n in exp.items():
        if s not in clean_sets:
            raise KeyError(f"clean_sets has no '{s}'. Keys: {list(clean_sets)}")
        im, mk = clean_sets[s]
        if len(im) != n or len(mk) != n:
            raise ValueError(f"{s}: {len(im)} images / {len(mk)} masks, expected {n} "
                             "(see the dataset table of the paper). Rebuild clean_sets as in the notebook.")


def _store_images(imgs, masks):
    """Images and masks at QUAL_SIZE for the qualitative figure (JPEG / packed bits)."""
    jpgs, gts = [], []
    for ip, mp in zip(imgs, masks):
        im = Image.open(ip).convert('RGB').resize((QUAL_SIZE, QUAL_SIZE), Image.BILINEAR)
        b = io.BytesIO(); im.save(b, format='JPEG', quality=90); jpgs.append(np.frombuffer(b.getvalue(), np.uint8))
        g = np.array(Image.open(mp).convert('L').resize((QUAL_SIZE, QUAL_SIZE), Image.NEAREST)) > 127
        gts.append(np.packbits(g))
    return jpgs, gts


def run_domain(domain, clean_sets, ckpt_dir, out_subdir='extended_eval', groups=None):
    import pandas as pd
    _check_sets(domain, clean_sets)
    out = os.path.join(ckpt_dir, out_subdir); os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, f'validation_{domain}.txt')
    logf = open(log_path, 'w')
    def log(msg):
        print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

    sets = SETS[domain]
    regime = {s: 'ID' for s in sets['ID']} | {s: 'OOD' for s in sets['OOD']}
    runs = runs_for(domain)
    if groups: runs = [r for r in runs if r['group'] in groups]
    missing = [r['tag'] for r in runs if not os.path.exists(os.path.join(ckpt_dir, r['tag'] + '.pth'))]
    if missing:
        raise FileNotFoundError(f'missing checkpoints in {ckpt_dir}: {missing}')
    checks = checks_for(domain)
    log(f'{domain}: {len(runs)} checkpoints, {sum(SETS[domain]["n"].values())} test images, '
        f'{datetime.datetime.now():%Y-%m-%d %H:%M}, torch {torch.__version__}, device {DEVICE}')
    log(f'NSD tolerances (fraction of width): {", ".join(f"{f:.5f}" for f in TOL_FRACTIONS)}')

    rows, means, qual = [], {}, {}
    M = lambda rid, s: means[(rid, s)]
    group_order = ['main', 'capacity', 'seeds', 'budget', 'datasize']
    done, worst_all = set(), 0.0
    csv_path = os.path.join(out, f'metrics_{domain}.csv')
    for g in group_order:
        grp = [r for r in runs if r['group'] == g]
        if not grp: continue
        for run in grp:
            t0 = time.time()
            model = load_checkpoint(run, ckpt_dir)
            keep = run['group'] == 'main'
            for s in sets['ID'] + sets['OOD']:
                im, mk = clean_sets[s]
                r_rows, preds = evaluate(model, im, mk, run['size'], keep_preds=keep)
                for rr in r_rows:
                    rr.update(domain=domain, run=run['id'], group=run['group'], tag=run['tag'],
                              arch=run['arch'], rank=run['rank'] or '', seed=run['seed'],
                              size=run['size'], set=s, regime=regime[s])
                rows += r_rows
                means[(run['id'], s)] = float(np.mean([rr['dice'] for rr in r_rows]))
                if keep: qual[f"pred__{run['id']}__{s}"] = np.array(preds, dtype=object)
            del model; torch.cuda.empty_cache()
            log(f"{run['id']:16s} {run['tag']:40s} {run['size']}px  {time.time() - t0:5.1f}s  " +
                '  '.join(f"{s} {means[(run['id'], s)]:.4f}" for s in sets['ID'] + sets['OOD']))
        done.add(g)
        log(f'-- checks after group "{g}"')
        worst, fails = run_checks([c for c in checks if c[1] == g], done, M, log)
        worst_all = max(worst_all, worst)
        if fails:
            pd.DataFrame(rows).to_csv(csv_path.replace('.csv', '_PARTIAL.csv'), index=False)
            log(f'STOPPED: {len(fails)} check(s) outside tolerance in group "{g}". '
                f'Largest deviation so far {worst_all:.4f}. Partial results saved; nothing else written.')
            logf.close()
            raise RuntimeError(f'validation failed: {fails}')

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    for s in sets['ID'] + sets['OOD']:
        jpgs, gts = _store_images(*clean_sets[s])
        qual[f'img__{s}'] = np.array(jpgs, dtype=object); qual[f'gt__{s}'] = np.array(gts, dtype=object)
        qual[f'names__{s}'] = np.array([os.path.basename(str(p)) for p in clean_sets[s][0]])
    np.savez_compressed(os.path.join(out, f'qualitative_{domain}.npz'), **qual)
    n_pe, n_ge = int(df.pred_empty.sum()), int(df.gt_empty.sum())
    log(f'ALL CHECKS PASSED. Largest |recomputed - reported| = {worst_all:.4f}')
    log(f'empty predictions: {n_pe} rows; empty ground truth after resizing: {n_ge} rows')
    log(f'wrote {csv_path} ({len(df)} rows)')
    logf.close()
    return df
