# Checks the BUSI duplicate removal. Requires `H` from the deduplication cell of breast.ipynb.
# Prints how many images are grouped as duplicates at thresholds 3, 5 and 8,
# and whether any duplicate pair at threshold 5 is also an identical file (md5)
# or an identical image (pixels).
import hashlib, itertools, re, numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image

SRC = Path('/content/data/busi/Dataset_BUSI_with_GT')
ph = H['BUSI']

def groups(thr):
    par = {n: n for n in ph}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in itertools.combinations(ph, 2):
        if ph[a] - ph[b] <= thr: par[find(a)] = find(b)
    g = defaultdict(list)
    for n in ph: g[find(n)].append(n)
    return [sorted(v) for v in g.values() if len(v) > 1]

print(f"BUSI lesion images: {len(ph)}")
for thr in [3, 5, 8]:
    m = groups(thr); drop = sum(len(v) - 1 for v in m)
    print(f"THR={thr}: {len(m)} groups, {sum(map(len, m))} images in groups, "
          f"{drop} dropped ({100*drop/len(ph):.1f}%)")

def raw(name):                        # benign_12.png -> Dataset_BUSI_with_GT/benign/benign (12).png
    cls, num = re.match(r'(benign|malignant)_(\d+)\.png', name).groups()
    return SRC / cls / f'{cls} ({num}).png'

m5 = groups(5)
md5_groups = pix_groups = 0
for g in m5:
    md5 = [hashlib.md5(raw(n).read_bytes()).hexdigest() for n in g]
    arr = [np.asarray(Image.open(raw(n)).convert('L')) for n in g]
    md5_groups += len(set(md5)) < len(md5)
    pix_groups += any(a.shape == b.shape and (a == b).all() for a, b in itertools.combinations(arr, 2))
print(f"THR=5 groups containing byte-identical source files (md5 would catch): {md5_groups} / {len(m5)}")
print(f"THR=5 groups containing pixel-identical images: {pix_groups} / {len(m5)}")
