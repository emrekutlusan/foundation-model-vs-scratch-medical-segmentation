r"""All generated tables, one consistent style:
booktabs rules only, no vertical rules, no scaling; tabular* spread to the full
text width (table*) or column width (table) with \extracolsep{\fill};
\footnotesize throughout (\scriptsize only for the full-page capacity table);
domains as italic group rows; bold = best value within a group."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import sys, io, contextlib
sys.path.insert(0, 'analysis')
from load import *
with contextlib.redirect_stdout(io.StringIO()):
    from make_results import per_image, MODELS, HIGHER, REPORTED
from make_capacity import CAP, PARAMS, REP as CAPREP, SETORD
from scipy.stats import wilcoxon
OUT = 'outputs/tables'
SHORT = {MODELS[0]: 'Plain U-Net', MODELS[1]: 'ResU-Net', MODELS[2]: 'LoRA $r{=}32$'}
SETLAB = {'Kvasir': 'Kvasir (in)', 'CVC-ClinicDB': 'CVC-ClinicDB (in)', 'BUSI-val': 'BUSI-val (in)'}
def B(t, on): return r'\textbf{' + t + '}' if on else t
def f3(v): return f'{v:.3f}'
def sgn(v, d=3): return f'${v:+.{d}f}$'

def env(star, caption, label, colspec, header, body, size=r'\footnotesize', place='tp', extra='', maxh=None):
    # maxh: optional macro name; the table is scaled down only if taller than it (used for the full-page table
    # in the single-column preprint, where \CapTabMax is set below \textheight; in two columns it is \textheight).
    width = r'\textwidth' if star else r'\columnwidth'
    e = 'table*' if star else 'table'
    pre = (f'\\providecommand{{{maxh}}}{{\\textheight}}\n\\begin{{adjustbox}}{{max totalheight={maxh}}}\n' if maxh else '')
    post = ('\\end{adjustbox}\n' if maxh else '')
    return (f'\\begin{{{e}}}[{place}]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n{extra}{pre}'
            f'\\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}\n\\toprule\n{header}\n\\midrule\n'
            f'{body}\n\\bottomrule\n\\end{{tabular*}}\n{post}\\end{{{e}}}\n')

def group(title, ncol, first=False):
    return ('' if first else '\\midrule\n') + f'\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n'

# ------------------------------------------------------------------ Main results, six metrics
mets = ['dice', 'iou', 'sensitivity', 'precision', 'hd95_norm', 'assd_norm', 'nsd_1']
body = ''
for gi, dom in enumerate(['polyp', 'breast', 'skin']):
    body += group(dom.capitalize(), 11, gi == 0)
    for si, s in enumerate(SETS[dom]):
        vals = {m: {M: per_image(dom, M, s, m).mean() for M in MODELS} for m in mets}
        emp = {M: per_image(dom, M, s, 'pred_empty').sum() for M in MODELS}
        n = len(per_image(dom, MODELS[0], s, 'dice')); rep = REPORTED[s]
        for k, M in enumerate(MODELS):
            row = [SETLAB.get(s, s) if k == 0 else '', str(n) if k == 0 else '', SHORT[M]]
            row.append(B(rep[k], rep[k] == max(rep, key=float)))
            for m in mets[1:]:
                best = (max if HIGHER[m] else min)(vals[m].values())
                row.append(B(f3(vals[m][M]), abs(vals[m][M] - best) < 1e-12))
            e = emp[M]; row.append(('%d' % e) if float(e).is_integer() else ('%.1f' % e))
            body += ' & '.join(row) + ' \\\\\n'
        if si < len(SETS[dom]) - 1: body += '\\addlinespace[2pt]\n'
cap = (r"Six metrics by model and test set. DSC, IoU, sensitivity (Sens), precision (Prec) and NSD: higher is better; HD95 and ASSD are divided by the image diagonal (Section~\ref{sec:eval}): lower is better. NSD at a tolerance of 2/224 of the image width. $\varnothing$: images with an empty prediction. Resolution 224\,px in polyp and breast; in skin 392\,px for DINOv2 and 384\,px for the U-Nets. Task-specific models trained for 300 epochs in polyp and breast and 100 in skin; DINOv2 for 40 epochs in polyp and breast and 20 in skin, single-seed except skin DINOv2, a three-seed mean (its $\varnothing$ is the mean per seed). (in) marks in-domain sets; BUSI-val also served for model selection. Asterisks mark skin sets drawn from the same archive as the training data; all CVC-T images also occur in CVC-ColonDB (Section~\ref{sec:datasets}). DSC values are those of the original evaluation, which the re-evaluation that produced the other metrics reproduces to within 0.0007 (Section~\ref{sec:eval}). Bold marks the best value for each test set.")
hdr = r'Test set & $n$ & Model & DSC & IoU & Sens & Prec & HD95 & ASSD & NSD & $\varnothing$ \\'
open(f'{OUT}/tab_main.tex', 'w').write(env(True, cap, 'tab:main', 'llrcccccccc' .replace('llrcccccccc', 'lrlcccccccr'), hdr, body.rstrip('\n')))

# ------------------------------------------------------------------ Empty predictions
cfgs = [('Plain U-Net', {'polyp': 'plain', 'breast': 'plain', 'skin': 'plain'}),
        ('ResU-Net', {'polyp': 'resunet', 'breast': 'resunet', 'skin': 'resunet'}),
        None,
        ('Frozen', {'polyp': 'frozen_e40', 'breast': 'frozen_e40', 'skin': None}),
        ('BitFit', {'polyp': 'bitfit_e40', 'breast': 'bitfit_e40', 'skin': 'bitfit'}),
        ('LoRA $r{=}4$', {'polyp': 'lora4_e40', 'breast': 'lora4_e40', 'skin': 'lora4'}),
        ('LoRA $r{=}8$', {'polyp': 'lora8_e40', 'breast': 'lora8_e40', 'skin': 'lora8'}),
        ('LoRA $r{=}16$', {'polyp': 'lora16_e40', 'breast': 'lora16_e40', 'skin': 'lora16'}),
        ('LoRA $r{=}32$', {'polyp': 'lora32_e40', 'breast': 'lora32_e40', 'skin': 'lora32_s42'}),
        ('Full FT', {'polyp': 'full_e40', 'breast': 'full_e40', 'skin': 'full_s42'}),
        None,
        ('LoRA $r{=}32$, 10\\% data', {'polyp': 'lora32_f10', 'breast': None, 'skin': None}),
        ('LoRA $r{=}32$, 25\\% data', {'polyp': 'lora32_f25', 'breast': 'lora32_f25', 'skin': None}),
        ('LoRA $r{=}32$, 50\\% data', {'polyp': 'lora32_f50', 'breast': 'lora32_f50', 'skin': None}),
        ('ResU-Net, 10\\% data', {'polyp': 'resunet_f10', 'breast': None, 'skin': None}),
        ('ResU-Net, 25\\% data', {'polyp': 'resunet_f25', 'breast': 'resunet_f25', 'skin': None}),
        ('ResU-Net, 50\\% data', {'polyp': 'resunet_f50', 'breast': 'resunet_f50', 'skin': None})]
cols = [('polyp', s) for s in SETS['polyp']] + [('breast', s) for s in SETS['breast']] + [('skin', s) for s in SETS['skin']]
body = 'Test images & ' + ' & '.join(str(int((ALL[(ALL.domain == d) & (ALL.set == s) & (ALL.run == 'plain')]).shape[0])) for d, s in cols) + ' \\\\\n'
groups = ['Task-specific models (Table~\\ref{tab:main})', 'Capacity axis (Table~\\ref{tab:capacity})', 'Data-size sweep (Table~\\ref{tab:datasize})']
gi = 0
for c in [None] + cfgs:
    if c is None:
        body += '\\midrule\n' + f'\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{{groups[gi]}}}}} \\\\\n'; gi += 1; continue
    name, spec = c; r = [name]
    for d, s in cols:
        run = spec[d]
        r.append('--' if run is None else str(int(ALL[(ALL.domain == d) & (ALL.set == s) & (ALL.run == run)].pred_empty.sum())))
    body += ' & '.join(r) + ' \\\\\n'
hdr = (r' & \multicolumn{5}{c}{Polyp} & \multicolumn{3}{c}{Breast} & \multicolumn{3}{c}{Skin} \\' + '\n'
       r'\cmidrule(lr){2-6}\cmidrule(lr){7-9}\cmidrule(l){10-12}' + '\n'
       r'Configuration & Kvasir & ClinicDB & ColonDB & ETIS & CVC-T & BUSI-val & BrEaST & UCLM & 2016* & 2017* & PH2 \\')
cap = (r"Images on which a model predicts no lesion at all, although every reference mask contains one. Task-specific models as in Table~\ref{tab:main}; capacity-axis configurations at 40 epochs in polyp and breast and 20 in skin, seed 42; data-size runs as in Section~\ref{sec:datasize}. --: configuration not run.")
open(f'{OUT}/tab_empty.tex', 'w').write(env(True, cap, 'tab:empty', 'l' + 'c' * 11, hdr, body.rstrip('\n'),
                                               extra='\\setlength{\\tabcolsep}{2pt}\n'))  # fits the single-column width

# ------------------------------------------------------------------ Lesion size
body = ''
for gi_, dom in enumerate(['polyp', 'breast', 'skin']):
    ref = ALL[(ALL.domain == dom) & (ALL.run == 'plain') & ALL.set.isin(OOD[dom])][['set', 'image', 'gt_frac']].copy()
    q = ref.gt_frac.quantile([1 / 3, 2 / 3]).values
    ref['st'] = pd.cut(ref.gt_frac, [-1, q[0], q[1], 2], labels=['Small', 'Medium', 'Large'])
    for st in ['Small', 'Medium', 'Large']:
        keys = ref[ref.st == st]
        vals = {}
        for M in MODELS:
            v = []
            for s in OOD[dom]:
                pi = per_image(dom, M, s, 'dice'); v += list(pi.loc[keys[keys.set == s].image].values)
            vals[M] = np.mean(v)
        best = max(vals.values())
        r = [dom.capitalize() if st == 'Small' else '', st, f'{100*keys.gt_frac.min():.1f}--{100*keys.gt_frac.max():.1f}', str(len(keys))]
        r += [B(f3(vals[M]), abs(vals[M] - best) < 1e-12) for M in MODELS]
        r.append(sgn(vals[MODELS[2]] - max(vals[MODELS[0]], vals[MODELS[1]])))
        body += ' & '.join(r) + ' \\\\\n'
    if gi_ < 2: body += '\\midrule\n'
hdr = r'Domain & Lesion & Area (\%) & $n$ & Plain & ResU & LoRA & $\Delta$ \\'
cap = (r"Out-of-distribution DSC by lesion size. Images are split into terciles of reference lesion area (per cent of the image) within each domain, pooled over its out-of-distribution sets. Plain: plain U-Net; ResU: ResU-Net; LoRA: DINOv2 + LoRA $r{=}32$. $\Delta$: LoRA minus the better task-specific model. Models and budgets as in Table~\ref{tab:main}.")
open(f'{OUT}/tab_size.tex', 'w').write(env(False, cap, 'tab:size', 'llcrcccc', hdr, body.rstrip('\n'), place='tbp', extra='\\setlength{\\tabcolsep}{3pt}\n'))

# ------------------------------------------------------------------ Skin boundary quality at two resolutions
body = ''
for gi_, (lab, l, r_) in enumerate([('224\\,px for both models (data-size sweep, 100\\%)', ['lora32_224_f100'], ['resunet_224_f100']),
                                   ('Native resolution, 392\\,px (L) and 384\\,px (R) (Table~\\ref{tab:main})', ['lora32_s42', 'lora32_s123', 'lora32_s2024'], ['resunet'])]):
    body += group(lab, 9, gi_ == 0)
    for s in OOD['skin']:
        d = ALL[(ALL.domain == 'skin') & (ALL.set == s)]
        a = d[d.run.isin(l)].groupby('image')[['dice', 'nsd_1', 'nsd_2', 'hd95_norm']].mean().mean()
        b = d[d.run.isin(r_)].groupby('image')[['dice', 'nsd_1', 'nsd_2', 'hd95_norm']].mean().mean()
        rr = [s]
        for m in ['dice', 'nsd_1', 'nsd_2', 'hd95_norm']:
            la = (a[m] >= b[m]) if HIGHER[m] else (a[m] <= b[m])
            rr += [B(f3(a[m]), la), B(f3(b[m]), not la)]
        body += ' & '.join(rr) + ' \\\\\n'
hdr = (r' & \multicolumn{2}{c}{DSC} & \multicolumn{2}{c}{NSD (2/224)} & \multicolumn{2}{c}{NSD (4/224)} & \multicolumn{2}{c}{HD95} \\' + '\n'
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}' + '\n'
       r'Test set & L & R & L & R & L & R & L & R \\')
cap = (r"Boundary quality in skin at two resolutions: DINOv2 + LoRA $r{=}32$ (L) against ResU-Net (R). The 224\,px runs are the 100\% fraction of the data-size sweep, in which budgets were scaled (LoRA 20 epochs, ResU-Net 80); the native runs are those of Table~\ref{tab:main} (LoRA three-seed mean, 20 epochs; ResU-Net 100 epochs), so the comparison changes resolution together with these budgets rather than resolution alone. HD95 divided by the image diagonal. Bold marks the better model in each pair.")
open(f'{OUT}/tab_res.tex', 'w').write(env(True, cap, 'tab:res', 'l' + 'c' * 8, hdr, body.rstrip('\n')))

# ------------------------------------------------------------------ Capacity axis (full page)
cm = ['dice', 'iou', 'sensitivity', 'precision', 'hd95_norm', 'assd_norm', 'nsd_1', 'nsd_2']
body = ''
for gi_, dom in enumerate(['polyp', 'breast', 'skin']):
    body += group(dom.capitalize(), 11, gi_ == 0)
    for si, s in enumerate(SETORD[dom]):
        runs = [r for _, r in CAP[dom]]
        g = ALL[(ALL.domain == dom) & (ALL.set == s)].groupby('run')[cm].mean().loc[runs]
        dsc = CAPREP.get(s, [f'{v:.4f}' for v in g.dice.values]); db = max(dsc, key=float)
        best = {m: (g[m].max() if HIGHER[m] else g[m].min()) for m in cm}
        for k, (lab, r) in enumerate(CAP[dom]):
            row = [SETLAB.get(s, s) if k == 0 else '', lab, PARAMS[lab], B(dsc[k], dsc[k] == db)]
            row += [B(f3(g.loc[r, m]), abs(g.loc[r, m] - best[m]) < 1e-12) for m in cm[1:]]
            body += ' & '.join(row) + ' \\\\\n'
        if si < len(SETORD[dom]) - 1: body += '\\addlinespace[2pt]\n'
hdr = r'Test set & Configuration & Trainable & DSC & IoU & Sens & Prec & HD95 & ASSD & NSD$_{2}$ & NSD$_{4}$ \\'
cap = (r"Capacity axis by test set in all metrics. Polyp and breast at 40 epochs, skin at 20 to match its rank sweep, all seed 42. The skin frozen configuration is omitted: its checkpoint was trained under an earlier protocol and is not comparable. DSC values are those of the original evaluation, except on CVC-ClinicDB, which was not reported there and is recomputed. HD95 and ASSD are divided by the image diagonal (lower is better); NSD$_{2}$ and NSD$_{4}$ at 2/224 and 4/224 of the image width. BitFit's BUSI-val value is its own selection criterion, not an independent result. (in) marks in-domain sets. Bold marks the best configuration for each test set and metric.")
open(f'{OUT}/tab_capacity.tex', 'w').write(env(True, cap, 'tab:capacity', 'llrcccccccc', hdr, body.rstrip('\n'), size='\\scriptsize', place='p',
                                                 extra='\\renewcommand{\\arraystretch}{0.94}\n\\setlength{\\tabcolsep}{2pt}\n', maxh='\\CapTabMax'))
# ------------------------------------------------------------------ Adaptation strategies
FULL = 22056961
strat = [('Frozen', 385, 'None'), ('BitFit', 51841, 'All bias terms'),
         ('LoRA $r{=}4$', 74113, 'Rank-4 factors on Q and V'), ('LoRA $r{=}8$', 147841, 'Rank-8 factors on Q and V'),
         ('LoRA $r{=}16$', 295297, 'Rank-16 factors on Q and V'), ('LoRA $r{=}32$', 590209, 'Rank-32 factors on Q and V'),
         ('Full FT', FULL, 'All weights')]
for lab, n, _ in strat[2:6]:                      # LoRA counts: 12 blocks x (Q, V) x r x (384 + 384) + decoder
    r = int(lab.split('=}')[1].rstrip('$'))
    assert n == 12 * 2 * r * (384 + 384) + 385, lab
body = '\n'.join(f"{lab} & {n:,} & {('%.3f' if n < 1000 else '%.2f') % (100 * n / FULL) if n != FULL else '100'} & {u} \\\\"
                 for lab, n, u in strat)
cap = (r"Adaptation strategies on the capacity axis. Every configuration also trains the 385-parameter linear decoder, "
       r"which is included in the trainable count; the percentage is relative to full fine-tuning. Q and V: query and value "
       r"projections of every attention block.")
open(f'{OUT}/tab_strategies.tex', 'w').write(env(False, cap, 'tab:strategies', 'lrrl',
     r'Configuration & Trainable & \% of full FT & Encoder parameters updated \\', body))

print('tables written to outputs/tables')
