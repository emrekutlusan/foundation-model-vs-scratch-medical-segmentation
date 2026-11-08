# Content-level overlap check between CVC-T (CVC-300) and CVC-ColonDB.
# Requires `clean_sets` from the data-preparation cells of polyp.ipynb.
import hashlib, numpy as np, os
from PIL import Image
def pix_hash(p, mode):
    a = np.array(Image.open(p).convert(mode))
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()
def file_hash(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

t_im, t_mk = clean_sets['CVC-T']; c_im, c_mk = clean_sets['CVC-ColonDB']
print('resolved paths')
print('  CVC-T       image dir:', os.path.realpath(os.path.dirname(t_im[0])), '| n =', len(t_im))
print('  CVC-ColonDB image dir:', os.path.realpath(os.path.dirname(c_im[0])), '| n =', len(c_im))
print('  CVC-T       mask dir :', os.path.realpath(os.path.dirname(t_mk[0])))
print('  CVC-ColonDB mask dir :', os.path.realpath(os.path.dirname(c_mk[0])))
print('  same image directory:', os.path.realpath(os.path.dirname(t_im[0])) == os.path.realpath(os.path.dirname(c_im[0])))

for kind, T, C, mode in [('images', t_im, c_im, 'RGB'), ('masks', t_mk, c_mk, 'L')]:
    cp = {}; cf = {}
    for p in C: cp.setdefault(pix_hash(p, mode), []).append(os.path.basename(p)); cf.setdefault(file_hash(p), []).append(os.path.basename(p))
    pix = [os.path.basename(p) for p in T if pix_hash(p, mode) in cp]
    byt = [os.path.basename(p) for p in T if file_hash(p) in cf]
    same_name = sum(1 for p in T if pix_hash(p, mode) in cp and os.path.basename(p) in cp[pix_hash(p, mode)])
    print(f'{kind}: {len(T)} CVC-T files | pixel-identical to a ColonDB file: {len(pix)} '
          f'(of which under the same file name: {same_name}) | byte-identical files: {len(byt)}')
    miss = [os.path.basename(p) for p in T if pix_hash(p, mode) not in cp]
    if miss: print('   no pixel-identical ColonDB counterpart:', miss[:10])
