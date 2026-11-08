"""Qualitative comparison figure. Cases are chosen by a fixed rule, not by eye: within each
domain's pooled out-of-distribution images, the image at the 90th and at the 10th percentile
of the paired difference DSC(LoRA r=32) - DSC(plain U-Net)."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import sys, io; sys.path.insert(0, 'analysis')
from load import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
RUN = {'polyp': {'Plain U-Net': ('plain', 224), 'ResU-Net': ('resunet', 224), 'LoRA r=32': ('lora32_e40', 224)},
       'breast': {'Plain U-Net': ('plain', 224), 'ResU-Net': ('resunet', 224), 'LoRA r=32': ('lora32_e40', 224)},
       'skin': {'Plain U-Net': ('plain', 384), 'ResU-Net': ('resunet', 384), 'LoRA r=32': ('lora32_s42', 392)}}
COL = {'Plain U-Net': '#eb6834', 'ResU-Net': '#1baf7a', 'LoRA r=32': '#2a78d6'}
Q = 256
def unpack(b, n): return np.unpackbits(np.asarray(b, dtype=np.uint8))[:n * n].reshape(n, n).astype(bool)
def resize(m): return np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((Q, Q), Image.NEAREST)) > 127
cases = []
for dom in ['polyp', 'breast', 'skin']:
    d = ALL[(ALL.domain == dom) & ALL.set.isin(OOD[dom])]
    a = d[d.run == RUN[dom]['LoRA r=32'][0]].set_index(['set', 'image']).dice
    b = d[d.run == 'plain'].set_index(['set', 'image']).dice
    diff = (a - b.loc[a.index]).sort_values()
    for lab, q in [('90th percentile', 0.9), ('10th percentile', 0.1)]:
        target = diff.quantile(q); key = (diff - target).abs().idxmin()
        cases.append((dom, lab, key, float(diff[key])))
z = {d: np.load(f'qual/{d}.npz', allow_pickle=True) for d in ['polyp', 'breast', 'skin']}
fig, axes = plt.subplots(len(cases), 4, figsize=(6.2, 1.5 * len(cases) + 0.2))
for r, (dom, lab, (s, name), dd) in enumerate(cases):
    names = list(z[dom][f'names__{s}']); i = names.index(name)
    img = np.array(Image.open(io.BytesIO(z[dom][f'img__{s}'][i].tobytes())).convert('RGB'))
    gt = unpack(z[dom][f'gt__{s}'][i], Q)
    gtb = gt ^ ndimage.binary_erosion(gt, iterations=2)
    ax = axes[r, 0]; ax.imshow(img); ov = np.zeros((Q, Q, 4)); ov[gtb] = (1, 0.83, 0, 1); ax.imshow(ov)
    ax.set_ylabel(f'{dom.capitalize()}\n{s}\n({lab})', fontsize=7)
    for c, M in enumerate(['Plain U-Net', 'ResU-Net', 'LoRA r=32']):
        run, n = RUN[dom][M]
        p = resize(unpack(z[dom][f'pred__{run}__{s}'][i], n))
        dsc = float(ALL[(ALL.domain == dom) & (ALL.set == s) & (ALL.run == run) & (ALL.image == name)].dice.iloc[0])
        ax = axes[r, c + 1]; ax.imshow(img)
        ov = np.zeros((Q, Q, 4)); rgb = matplotlib.colors.to_rgb(COL[M])
        ov[p] = (*rgb, 0.45); pb = p ^ ndimage.binary_erosion(p, iterations=2); ov[pb] = (*rgb, 1)
        ov[gtb] = (1, 0.83, 0, 1); ax.imshow(ov)
        ax.text(0.03, 0.04, f'DSC {dsc:.2f}' if p.any() else 'no lesion predicted', transform=ax.transAxes, fontsize=6.5,
                color='white', bbox=dict(boxstyle='round,pad=0.2', fc='black', ec='none', alpha=0.6))
for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
for c, t in enumerate(['Image and reference', 'Plain U-Net', 'ResU-Net', 'DINOv2 + LoRA r=32']):
    axes[0, c].set_title(t, fontsize=8)
fig.subplots_adjust(left=0.1, right=0.995, top=0.975, bottom=0.005, wspace=0.03, hspace=0.06)
fig.savefig('outputs/figs/fig_qual.pdf', dpi=300); fig.savefig('outputs/figs/fig_qual.png', dpi=110)
for c in cases: print(c)
