import numpy as np
import torch
from core import *

def test_group_size_and_identity():
    p=all_permutations(); assert p.shape==(120,5)
    ids=np.zeros((3,4,5),dtype=np.int64); ids[:]=np.arange(5)
    assert np.all(compose_numpy(ids)==np.arange(5))

def test_slow_fast_same_multiset_different_order():
    s=key_schedule('slow',200,100);f=key_schedule('fast',200,100);ds,df=schedule_digests(s),schedule_digests(f)
    assert ds['multiset_digest']==df['multiset_digest'];assert ds['temporal_digest']!=df['temporal_digest'];assert max_map_run(s)==100 and max_map_run(f)==1

def test_batch_key_identity():
    p=all_permutations();x1,y1=make_power_batch(3,'A',17,16,1.5,1729,31415,p);x2,y2=make_power_batch(3,'A',17,16,1.5,1729,31415,p)
    assert np.array_equal(x1,x2) and np.array_equal(y1,y2)

def test_map_heads_are_disjoint():
    a,b=map_orders(1729);assert head_overlap(a,b)==0

def test_persistence_h_extremes():
    k1=key_schedule('persistence',20,10,1);k5=key_schedule('persistence',20,10,5)
    assert max_map_run(k1)==1 and max_map_run(k5)==5;assert schedule_digests(k1)['multiset_digest']==schedule_digests(k5)['multiset_digest']

def test_model_forward():
    m=StateTrackingTransformer(d_model=32,layers=1,heads=4,ff_mult=2);out=m(torch.randint(0,5,(2,20)));assert out.shape==(2,5,5)
