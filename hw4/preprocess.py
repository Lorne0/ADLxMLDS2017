from utils import *
import pickle as pk

#tags = make_tags()
#imgs = load_image()
tags, imgs = make_better()
with open('./data/better_pretrained2.pk', 'wb') as fw:
    pk.dump(tags, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(imgs, fw, protocol=pk.HIGHEST_PROTOCOL)

