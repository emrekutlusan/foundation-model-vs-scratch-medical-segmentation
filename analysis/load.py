"""Per-image metrics of every evaluated model, the test sets of each domain, and the models of the main comparison."""
import os
for _d in ('outputs/tables', 'outputs/figs'): os.makedirs(_d, exist_ok=True)
import pandas as pd, numpy as np
P=pd.read_csv('results/metrics_polyp.csv')
B=pd.read_csv('results/metrics_breast.csv')
S=pd.read_csv('results/metrics_skin.csv')
ALL=pd.concat([P,B,S],ignore_index=True)
SETS={'polyp':['Kvasir','CVC-ClinicDB','CVC-ColonDB','ETIS','CVC-T'],'breast':['BUSI-val','BrEaST','BUS-UCLM'],'skin':['ISIC 2016*','ISIC 2017*','PH2']}
OOD={'polyp':['CVC-ColonDB','ETIS','CVC-T'],'breast':['BrEaST','BUS-UCLM'],'skin':['ISIC 2016*','ISIC 2017*','PH2']}
ID={'polyp':['Kvasir','CVC-ClinicDB'],'breast':['BUSI-val'],'skin':[]}
# models of the main comparison; skin LoRA is averaged over three seeds
def main_runs(dom):
    l={'polyp':['lora32_e40'],'breast':['lora32_e40'],'skin':['lora32_s42','lora32_s123','lora32_s2024']}[dom]
    return {'Plain U-Net':['plain'],'ResU-Net':['resunet'],'DINOv2 + LoRA $r{=}32$':l}
METRICS=['dice','iou','sensitivity','precision','hd95_norm','assd_norm','nsd_1']
