
from pathlib import Path

debug=False
submission=True
batch_size=2
device='cuda:0'
out='.'
image_size=128
arch='pretrained'
model_name='se_resnext50_32x4d'


datadir = Path('../input/bengaliai-cv19')
featherdir = Path('../input/bengaliaicv19feather')
outdir = Path('./out_SR128x64_8/')
