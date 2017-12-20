from utils import *
import pickle as pk

tags = make_tags()
imgs = load_image()
with open('./data/pretrained.pk', 'wb') as fw:
    pk.dump(tags, fw, protocol=pk.HIGHEST_PROTOCOL)
    pk.dump(imgs, fw, protocol=pk.HIGHEST_PROTOCOL)

