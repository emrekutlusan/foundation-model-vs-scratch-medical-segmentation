"""Capacity axis: configurations, trainable parameter counts and the DSC values reported in the paper,
checked against the recomputed per-image metrics. The table itself is written by tables.py."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import sys; sys.path.insert(0, 'analysis')
from load import *
HIGHER = {'dice': 1, 'iou': 1, 'sensitivity': 1, 'precision': 1, 'hd95_norm': 0, 'assd_norm': 0, 'nsd_1': 1, 'nsd_2': 1}
CAP = {'polyp': [('Frozen', 'frozen_e40'), ('BitFit', 'bitfit_e40'), ('LoRA $r{=}4$', 'lora4_e40'), ('LoRA $r{=}8$', 'lora8_e40'),
                 ('LoRA $r{=}16$', 'lora16_e40'), ('LoRA $r{=}32$', 'lora32_e40'), ('Full FT', 'full_e40')],
       'breast': [('Frozen', 'frozen_e40'), ('BitFit', 'bitfit_e40'), ('LoRA $r{=}4$', 'lora4_e40'), ('LoRA $r{=}8$', 'lora8_e40'),
                  ('LoRA $r{=}16$', 'lora16_e40'), ('LoRA $r{=}32$', 'lora32_e40'), ('Full FT', 'full_e40')],
       'skin': [('BitFit', 'bitfit'), ('LoRA $r{=}4$', 'lora4'), ('LoRA $r{=}8$', 'lora8'), ('LoRA $r{=}16$', 'lora16'),
                ('LoRA $r{=}32$', 'lora32_s42'), ('Full FT', 'full_s42')]}
PARAMS = {'Frozen': '385', 'BitFit': '51,841', 'LoRA $r{=}4$': '74,113', 'LoRA $r{=}8$': '147,841', 'LoRA $r{=}16$': '295,297',
          'LoRA $r{=}32$': '590,209', 'Full FT': '22,056,961'}
# DSC as originally reported, in CAP order; CVC-ClinicDB was not reported and is recomputed.
REP = {'CVC-ColonDB': ['0.5875', '0.7204', '0.7139', '0.7132', '0.7346', '0.7694', '0.7396'],
       'ETIS': ['0.4947', '0.6867', '0.6843', '0.6984', '0.6711', '0.6984', '0.7041'],
       'CVC-T': ['0.7612', '0.8634', '0.8675', '0.8520', '0.8749', '0.8665', '0.8739'],
       'Kvasir': ['0.8005', '0.8960', '0.8764', '0.8828', '0.8851', '0.8877', '0.8790'],
       'BrEaST': ['0.6099', '0.7337', '0.7165', '0.6957', '0.7179', '0.7133', '0.6941'],
       'BUS-UCLM': ['0.6660', '0.7449', '0.7087', '0.6979', '0.7226', '0.7046', '0.7130'],
       'BUSI-val': ['0.6493', '0.7853', '0.7572', '0.7751', '0.7950', '0.8029', '0.7796'],
       'ISIC 2016*': ['0.9238', '0.9236', '0.9242', '0.9233', '0.9246', '0.9215'],
       'ISIC 2017*': ['0.8684', '0.8710', '0.8635', '0.8629', '0.8763', '0.8717'],
       'PH2': ['0.9208', '0.9205', '0.9289', '0.9236', '0.9246', '0.9239']}
SETORD = {'polyp': ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS', 'CVC-T'], 'breast': ['BUSI-val', 'BrEaST', 'BUS-UCLM'],
          'skin': ['ISIC 2016*', 'ISIC 2017*', 'PH2']}
# reported DSC must match the recomputed values
for dom in SETORD:
    for s in SETORD[dom]:
        if s not in REP: continue
        g = ALL[(ALL.domain == dom) & (ALL.set == s)].groupby('run').dice.mean()
        dev = max(abs(float(v) - g[r]) for v, (_, r) in zip(REP[s], CAP[dom]))
        assert dev < 0.0011, (s, dev)

