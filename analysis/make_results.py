"""Shared definitions for the analysis, the summary numbers quoted in the text, and the
regime and per-image distribution figures."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import sys; sys.path.insert(0, 'analysis')
from load import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUTF = 'outputs/figs'
MODELS = ['Plain U-Net', 'ResU-Net', 'DINOv2 + LoRA $r{=}32$']
SHORT = {'Plain U-Net': 'Plain U-Net', 'ResU-Net': 'ResU-Net', 'DINOv2 + LoRA $r{=}32$': 'LoRA $r{=}32$'}
COL = {'Plain U-Net': '#eb6834', 'ResU-Net': '#1baf7a', 'DINOv2 + LoRA $r{=}32$': '#2a78d6'}
HIGHER = {'dice': 1, 'iou': 1, 'sensitivity': 1, 'precision': 1, 'nsd_1': 1, 'nsd_2': 1, 'hd95_norm': 0, 'assd_norm': 0, 'hd95_px': 0, 'assd_px': 0}
SETLAB = {'Kvasir': 'Kvasir (in)', 'CVC-ClinicDB': 'ClinicDB (in)', 'BUSI-val': 'BUSI-val (in)'}
RES = {'polyp': '224', 'breast': '224', 'skin': '384/392'}

def per_image(dom, model, s, met):
    """Per-image values; skin LoRA averaged over its three seeds image by image."""
    runs = main_runs(dom)[model]
    d = ALL[(ALL.domain == dom) & (ALL.set == s) & ALL.run.isin(runs)]
    return d.groupby('image')[met].mean()

# DSC as originally reported for the three models of the main comparison (Plain U-Net, ResU-Net, LoRA r=32)
REPORTED = {'Kvasir': ('0.8804', '0.8611', '0.8877'), 'CVC-ClinicDB': ('0.8803', '0.8826', '0.8684'),
            'CVC-ColonDB': ('0.7382', '0.7240', '0.7694'), 'ETIS': ('0.6321', '0.6179', '0.6984'), 'CVC-T': ('0.8406', '0.7941', '0.8665'),
            'BUSI-val': ('0.8177', '0.8244', '0.8029'), 'BrEaST': ('0.6545', '0.6463', '0.7133'), 'BUS-UCLM': ('0.6320', '0.5981', '0.7046'),
            'ISIC 2016*': ('0.9123', '0.9084', '0.924'), 'ISIC 2017*': ('0.8336', '0.8356', '0.872'), 'PH2': ('0.9167', '0.9032', '0.925')}
mets = ['dice', 'iou', 'sensitivity', 'precision', 'hd95_norm', 'assd_norm', 'nsd_1']

# ---------------------------------------------------------------- numbers for the text
def ood_mean(dom, M, met):
    return np.mean([per_image(dom, M, s, met).mean() for s in OOD[dom]])
summary = {dom: {M: {m: ood_mean(dom, M, m) for m in mets} for M in MODELS} for dom in OOD}
win = {}
for dom in OOD:
    for M in MODELS[:2]:
        w = []
        for s in OOD[dom]:
            a = per_image(dom, MODELS[2], s, 'dice'); b = per_image(dom, M, s, 'dice')
            w.append((a > b + 1e-12).sum() / len(a))
        win[(dom, M)] = w
pd.to_pickle({'summary': summary, 'win': win}, 'results/summary.pkl')

# ---------------------------------------------------------------- Figure: regimes, four panels
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.linewidth': 0.6, 'xtick.color': '#333', 'ytick.color': '#333',
                     'legend.frameon': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw={'width_ratios': [2, 3, 3, 3]})
panels = [('(a) In-domain DSC', 'ID', 'dice', ['polyp', 'breast']), ('(b) OOD DSC', 'OOD', 'dice', ['polyp', 'breast', 'skin']),
          ('(c) OOD HD95 / diagonal', 'OOD', 'hd95_norm', ['polyp', 'breast', 'skin']),
          ('(d) OOD NSD', 'OOD', 'nsd_1', ['polyp', 'breast', 'skin'])]
w = 0.26
for ax, (title, reg, met, doms) in zip(axes, panels):
    for k, M in enumerate(MODELS):
        ys = [np.mean([per_image(d, M, s, met).mean() for s in (ID if reg == 'ID' else OOD)[d]]) for d in doms]
        ax.bar(np.arange(len(doms)) + (k - 1) * w, ys, w * 0.92, color=COL[M], label=SHORT[M].replace('$r{=}32$', 'r=32'), edgecolor='white', linewidth=0.6)
    ax.set_xticks(np.arange(len(doms))); ax.set_xticklabels([d.capitalize() for d in doms])
    ax.set_title(title, fontsize=8); ax.grid(axis='y', color='#e5e5e5', lw=0.5); ax.set_axisbelow(True)
    if met == 'dice': ax.set_ylim(0.5, 0.95)
    if met == 'nsd_1': ax.set_ylim(0.2, 0.75)
fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f'{OUTF}/fig_regimes.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure: per-image distributions
oodsets = [(d, s) for d in ['polyp', 'breast', 'skin'] for s in OOD[d]]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
for ax, (met, lab) in zip(axes, [('dice', 'DSC'), ('hd95_norm', 'HD95 / diagonal')]):
    pos = []
    for i, (d, s) in enumerate(oodsets):
        for k, M in enumerate(MODELS):
            v = per_image(d, M, s, met).dropna().values
            p = i + (k - 1) * 0.27
            bp = ax.boxplot(v, positions=[p], widths=0.22, patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', markersize=1.5, markerfacecolor=COL[M], markeredgecolor='none', alpha=0.5),
                            medianprops=dict(color='white', lw=1.0), whiskerprops=dict(color=COL[M], lw=0.7),
                            capprops=dict(color=COL[M], lw=0.7), boxprops=dict(facecolor=COL[M], edgecolor=COL[M]))
            ax.plot(p, v.mean(), marker='D', ms=2.2, color='black', zorder=5)
    ax.set_ylabel(lab); ax.grid(axis='y', color='#e5e5e5', lw=0.5); ax.set_axisbelow(True)
    for x in (2.5, 4.5): ax.axvline(x, color='#bbb', lw=0.6)
axes[1].set_xticks(range(len(oodsets))); axes[1].set_xticklabels([s for _, s in oodsets], fontsize=7.5)
axes[0].set_ylim(-0.02, 1.02)
axes[1].set_yscale('symlog', linthresh=0.01, linscale=0.3); axes[1].set_ylim(0, 1.05)
axes[1].set_yticks([0, 0.01, 0.03, 0.1, 0.3, 1]); axes[1].set_yticklabels(['0', '0.01', '0.03', '0.1', '0.3', '1'])
axes[1].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
for x, t in [(1, 'Polyp'), (3.5, 'Breast'), (6, 'Skin')]: axes[0].text(x, 1.06, t, ha='center', fontsize=8, transform=axes[0].get_xaxis_transform())
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[M]) for M in MODELS]
axes[1].legend(handles, [SHORT[M].replace('$r{=}32$', 'r=32') for M in MODELS], loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout()
fig.savefig(f'{OUTF}/fig_dist.pdf'); plt.close(fig)

