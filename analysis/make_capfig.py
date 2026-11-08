"""Capacity figure: out-of-distribution DSC against trainable parameters for each domain."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import sys; sys.path.insert(0, 'analysis')
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from make_capacity import CAP, REP
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = {'Frozen': 385, 'BitFit': 51841, 'LoRA $r{=}4$': 74113, 'LoRA $r{=}8$': 147841, 'LoRA $r{=}16$': 295297, 'LoRA $r{=}32$': 590209, 'Full FT': 22056961}
ID = {'polyp': ['Kvasir'], 'breast': ['BUSI-val'], 'skin': []}
OOD = {'polyp': ['CVC-ColonDB', 'ETIS', 'CVC-T'], 'breast': ['BrEaST', 'BUS-UCLM'], 'skin': ['ISIC 2016*', 'ISIC 2017*', 'PH2']}
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#555', 'axes.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=False)
for ax, dom in zip(axes, ['polyp', 'breast', 'skin']):
    labs = [l for l, _ in CAP[dom]]; x = np.array([N[l] for l in labs])
    for kind, sets, ls, mk in [('In domain', ID[dom], '-', 'o'), ('Out of distribution (mean)', OOD[dom], '--', 's')]:
        if not sets: continue
        y = np.mean([[float(v) for v in REP[s]] for s in sets], axis=0)
        ax.plot(x, y, ls, marker=mk, ms=3.5, lw=1.4, color='#2a78d6' if ls == '-' else '#eb6834', label=kind)
    ax.set_ylim(0.55, 0.95); ax.set_xlim(150, 6e7)
    ax.set_xticks([385, 51841, 590209, 22056961]); ax.set_xticklabels(['Frozen\n385', 'BitFit\n52K', 'LoRA r=32\n590K', 'Full FT\n22M'], fontsize=6.5)
    ax.minorticks_off()
    ax.set_xscale('log'); ax.set_title(dom.capitalize(), fontsize=8.5); ax.grid(color='#e5e5e5', lw=0.5); ax.set_axisbelow(True)
    ax.set_xlabel('Trainable parameters (log scale)')
axes[0].set_ylabel('DSC')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('outputs/figs/fig_capacity.pdf'); fig.savefig('outputs/figs/fig_capacity_preview.png', dpi=120)
print('ok')
