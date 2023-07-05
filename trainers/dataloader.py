import numpy as np
import torch
import pandas as pd
import json
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

preprocess = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(size=(224, 224)),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

CMP_NAMES = ['more', 'less', 'equal', 'It\'s', 'Change', '']

CLASS_NAMES = {
    "shoes": ['open', 'pointy', 'sporty', 'comfort'],
    "fashionIQ": ['dress', 'shirt', 'toptee'],
    "fashion200k": ['cocktail dresses', 'prom and formal dresses', 'gowns', 'maxi and long dresses',
                    'casual and day dresses', 'mini and short dresses', 'casual jackets', 'blazers and suit jackets',
                    'padded and down jackets', 'fur jackets', 'waistcoats and gilets', 'leather jackets',
                    'denim jackets', 'leggings', 'full length pants', 'cropped pants', 'skinny pants',
                    'wide-leg and palazzo pants', 'straight-leg pants', 'cargo pants', 'harem pants',
                    'mid length skirts', 'maxi skirts', 'knee length skirts', 'mini skirts', 'long sleeved tops',
                    't-shirts', 'blouses', 'shirts', 'short sleeve tops', 'sleeveless and tank tops'],
    "MIT": ['the state to' for _ in range(245)],
    "celebA": ['the attribute to' for _ in range(40)],
    "B2W": ['sameFamily', 'sameSpecies', 'visual', 'sameClass', 'sameGenus', 'sameOrder'],
}


class CelebAData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = np.asarray([4] * len(self.labels))
        random.seed(cfg.seed)
        if subset == 'test':
            with open(cfg.root + 'celebA/test_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'celebA/test_text.json', 'r') as f:
                texts = json.load(f)
        else:
            with open(cfg.root + 'celebA/train_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'celebA/train_text.json', 'r') as f:
                texts = json.load(f)

        if subset == 'train':
            counts = dict()
            seq, sel = list(range(len(self.labels))), [0] * len(self.labels)
            random.shuffle(seq)
            for j in seq:
                i = self.labels[j]
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel[j] = 1
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[k, ''] + [texts[str(i)][str(j)] + '.']
                         for i, j, k in zip(self.ref_index, self.tgt_index, self.labels)]
        self.ref_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[str(i)] for i in self.ref_index]
        self.tgt_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[str(i)] for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_name = self.ref_name[sent_id]
        tgt_name = self.tgt_name[sent_id]
        ref_index = self.ref_index[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id]
        compare = self.compare[sent_id]
        texts = self.tgt_text[sent_id]

        return ref_name, tgt_name, ref_index, tgt_index, label, compare, texts


class FashionIQData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = h5loader['compare']
        random.seed(cfg.seed)
        if subset == 'test':
            with open(cfg.root + 'fashionIQ/meta/test_images.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'fashionIQ/meta/test_text.json', 'r') as f:
                texts = json.load(f)
        else:
            with open(cfg.root + 'fashionIQ/meta/train_images.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'fashionIQ/meta/train_text.json', 'r') as f:
                texts = json.load(f)

        if subset == 'train':
            counts = dict()
            seq, sel = list(range(len(self.labels))), [0] * len(self.labels)
            random.shuffle(seq)
            for j in seq:
                i = self.labels[j]
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel[j] = 1
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[k, ''] + [t + '.' for t in texts[str(i)][str(j)]]
                         for i, j, k in zip(self.ref_index, self.tgt_index, self.labels)]
        self.ref_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[i] + '.jpg' for i in self.ref_index]
        self.tgt_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[i] + '.jpg' for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_name = self.ref_name[sent_id]
        tgt_name = self.tgt_name[sent_id]
        ref_index = self.ref_index[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id]
        compare = self.compare[sent_id]
        texts = self.tgt_text[sent_id]

        return ref_name, tgt_name, ref_index, tgt_index, label, compare, texts


class B2WData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = np.asarray([5] * len(self.labels))
        random.seed(cfg.seed)
        if subset == 'test':
            with open(cfg.root + 'B2W/test_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'B2W/test_text.json', 'r') as f:
                texts = json.load(f)
        else:
            with open(cfg.root + 'B2W/train_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'B2W/train_text.json', 'r') as f:
                texts = json.load(f)

        if subset == 'train':
            counts = dict()
            seq, sel = list(range(len(self.labels))), [0] * len(self.labels)
            random.shuffle(seq)
            for j in seq:
                i = self.labels[j]
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel[j] = 1
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[k, ''] + [texts[str(i)][str(j)]]
                         for i, j, k in zip(self.ref_index, self.tgt_index, self.labels)]
        self.ref_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[str(i)] for i in self.ref_index]
        self.tgt_name = [cfg.root + cfg.dataset_name + '/images/' + image_list[str(i)] for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_name = self.ref_name[sent_id]
        tgt_name = self.tgt_name[sent_id]
        ref_index = self.ref_index[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id]
        compare = self.compare[sent_id]
        texts = self.tgt_text[sent_id]

        return ref_name, tgt_name, ref_index, tgt_index, label, compare, texts


def cap_collate_fn(data):
    ref_name, tgt_name, ref_index, tgt_index, label, compare, texts = zip(*data)
    ref_features = []
    tgt_features = []
    ref_indexes = []
    tgt_indexes = []
    ref_names = []
    tgt_names = []
    labels = []
    compares = []

    for ref, tgt, ref_idx, tgt_idx, lab, cmp in zip(ref_name, tgt_name, ref_index, tgt_index, label, compare):
        try:
            ref_image = Image.open(ref).convert("RGB")
        except:
            continue
        try:
            tgt_image = Image.open(tgt).convert("RGB")
        except:
            continue
        ref_features.append(preprocess(ref_image))
        tgt_features.append(preprocess(tgt_image))
        ref_indexes.append(ref_idx)
        tgt_indexes.append(tgt_idx)
        ref_names.append(ref)
        tgt_names.append(tgt)
        labels.append(lab)
        compares.append(cmp)

    ref_features = torch.stack(ref_features)
    tgt_features = torch.stack(tgt_features)
    labels = np.asarray(labels).astype(np.int)
    compare = np.asarray(compares).astype(np.int)
    ref_indexes = np.asarray(ref_indexes).astype(np.int)
    tgt_indexes = np.asarray(tgt_indexes).astype(np.int)

    entry = {
        'ref_features': ref_features,
        'tgt_features': tgt_features,
        'ref_indexes': ref_indexes,
        'tgt_indexes': tgt_indexes,
        'ref_names': ref_names,
        'tgt_names': tgt_names,
        'labels': labels,
        'compare': compare,
        'tgt_text': texts
    }

    return entry


class ShoesData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = h5loader['compare']
        meta = pd.read_csv(cfg.root + cfg.dataset_name + '/meta-data.csv', header=0, index_col=None)
        self.category = [meta['Category'][i] + meta['SubCategory'][i] for i in self.tgt_index]
        cat2ind = list(set(self.category))
        cat2ind.sort()
        cat2ind = {c: i for i, c in enumerate(cat2ind)}

        if subset == 'train':
            counts = dict()
            sel = []
            for i in self.category:
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel.append(1)
                else:
                    sel.append(0)
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[cat2ind[meta['Category'][i] + meta['SubCategory'][i]]] +
                         [item[i] for key, item in meta.items()][:3] for i in self.tgt_index]
        self.ref_names = [cfg.root + cfg.dataset_name + '/images/' +
                          '.'.join(meta['CID'][i].split('-')) + '.jpg' for i in self.ref_index]
        self.tgt_names = [cfg.root + cfg.dataset_name + '/images/' +
                          '.'.join(meta['CID'][i].split('-')) + '.jpg' for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_name = self.ref_names[sent_id]
        tgt_name = self.tgt_names[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id] - 1
        compare = self.compare[sent_id] - 1
        texts = self.tgt_text[sent_id]

        return ref_name, tgt_name, tgt_index, label, compare, texts


class MITData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = np.asarray([3] * len(self.labels))
        if subset == 'test':
            with open(cfg.root + 'MIT/test_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'MIT/test_text.json', 'r') as f:
                texts = json.load(f)
        else:
            with open(cfg.root + 'MIT/train_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'MIT/train_text.json', 'r') as f:
                texts = json.load(f)

        if subset == 'train':
            counts = dict()
            sel = []
            for i in self.labels:
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel.append(1)
                else:
                    sel.append(0)
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[k, ''] + [texts[str(i)][str(j)][1] + '.']
                         for i, j, k in zip(self.ref_index, self.tgt_index, self.labels)]
        self.ref_index = [cfg.root + cfg.dataset_name + '/' + image_list[str(i)] for i in self.ref_index]
        self.tgt_index = [cfg.root + cfg.dataset_name + '/' + image_list[str(i)] for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_index = self.ref_index[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id]
        compare = self.compare[sent_id]
        texts = self.tgt_text[sent_id]

        return ref_index, tgt_index, label, compare, texts


class Fashion200kData(Dataset):
    def __init__(self, cfg, subset, h5loader):
        self.cfg = cfg
        self.subset = subset
        self.ref_index = h5loader['ref_index']
        self.tgt_index = h5loader['tgt_index']
        self.labels = h5loader['labels']
        self.compare = np.asarray([3] * len(self.labels))
        if subset == 'test':
            with open(cfg.root + 'fashion200k/test_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'fashion200k/test_text.json', 'r') as f:
                texts = json.load(f)
        else:
            with open(cfg.root + 'fashion200k/train_image.json', 'r') as f:
                image_list = json.load(f)
            with open(cfg.root + 'fashion200k/train_text.json', 'r') as f:
                texts = json.load(f)

        if subset == 'train':
            counts = dict()
            sel = []
            for i in self.labels:
                if i not in counts:
                    counts[i] = 0
                if counts[i] < int(cfg.n_shot):
                    sel.append(1)
                else:
                    sel.append(0)
                counts[i] += 1
            sel = np.asarray(sel)
            self.ref_index = self.ref_index[sel > 0]
            self.tgt_index = self.tgt_index[sel > 0]
            self.labels = self.labels[sel > 0]
            self.compare = self.compare[sel > 0]
        self.tgt_text = [[k, ''] + ['change ' + texts[str(i)][str(j)][0] + ' to ' + texts[str(i)][str(j)][1] + '.']
                         for i, j, k in zip(self.ref_index, self.tgt_index, self.labels)]
        self.ref_index = [cfg.root + cfg.dataset_name + '/' + image_list[str(i)] for i in self.ref_index]
        self.tgt_index = [cfg.root + cfg.dataset_name + '/' + image_list[str(i)] for i in self.tgt_index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sent_id):
        ref_index = self.ref_index[sent_id]
        tgt_index = self.tgt_index[sent_id]
        label = self.labels[sent_id]
        compare = self.compare[sent_id]
        texts = self.tgt_text[sent_id]

        return ref_index, tgt_index, label, compare, texts
