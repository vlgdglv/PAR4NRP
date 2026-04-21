import numpy as np, glob

files = sorted(glob.glob('/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes/imagenet384_codes_sharded/*.npz'))
print('shards:', len(files))                                                                                                                                                                                                                          
total = 0                                                                                                                                                                                                                                             
label_hist = np.zeros(1000, dtype=np.int64)                                                                                                                                                                                                           
for f in files:                                                                                                                                                                                                                                       
    z = np.load(f)                                                                                                                                                                                                                                    
    total += z['labels'].shape[0]
    c, n = np.unique(z['labels'], return_counts=True)                                                                                                                                                                                                 
    label_hist[c] += n                                                                                                                                                                                                                                
print('total imgs:', total)
print('classes covered:', (label_hist > 0).sum(), '/ 1000')                                                                                                                                                                                           
print('min/max per class:', label_hist.min(), label_hist.max())                                                                                                                                                                                       
print('codes dtype/shape/one row:', z['codes'].dtype, z['codes'].shape)
print('token id range:', z['codes'].min(), z['codes'].max())                                                                                                                                                                                          
