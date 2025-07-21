
import onnxruntime as ort
import time
from strhub.models.utils import load_from_checkpoint
from strhub.data.module import SceneTextDataModule
from strhub.data.utils import CharsetAdapter

from PIL import Image
from typing import List, Tuple
from torch import Tensor
import numpy as np
import torch
from tqdm import tqdm

import io
import unicodedata
from typing import Callable, Optional
from PIL import Image
import lmdb
import random


class LmdbDataset:
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = [] 
        self.paths = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels(
            charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim
        )

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False, readahead=False, meminit=False, lock=False
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
            

                # Normally, whitespace is removed from the labels.
                if remove_whitespace:
                    label = ''.join(label.split())
                # Normalize unicode composites (if any) and convert to compatible ASCII characters
                # print('unicode: ', normalize_unicode)
                if normalize_unicode:
                    label = unicodedata.normalize('NFKD', label).encode('ascii', 'ignore').decode()
                # Filter by length before removing unsupported characters. The original label might be too long.
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                # Filter images that are too small.
                if min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        try:
            index = self.filtered_index_list[index]
            img_key = f'image-{index:09d}'.encode()
            label_key = f'label-{index:09d}'.encode()
            path_key = f'path-{index:09d}'.encode()
            with self.env.begin() as txn:
                imgbuf = txn.get(img_key)
                label = txn.get(label_key).decode()
                path = txn.get(path_key)

            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            return img, label, path
        except Exception as e:
            print(f'Error reading index {index}: {e}')
            return None, None, None
    

class TokenDecoder:
    def __init__(self):
        self.specials_first = ('[E]',)
        self.specials_last = ('[B]', '[P]')
        self.charset = tuple(i for i in "ཡིད་གསུམཕརབཐོལའྱཤ །ྡྗེཆངཀཉནཔཟླཙ༑ཏཁྒྣྲ༌ྔྷྭཞྐྙཅྟྤྩཚ༈ཛཇྫྨྦཨཱཾཧཎ༄༅ཝྕ༔ཥྜྋཌ༼ཊ༴ྪཻཿ༽ༀྚཀྵ࿄	༎༉࿚ཽ྾༒ྺ༐࿙༜༆࿅ྀ྇ྰྠྃྵཪྴྛྞ༵༞ྻ༸྅༷ྶྥྑཋ༏྄༝࿐༹ྂ࿉༙༗ཬ")
        self.itos = self.specials_first + self.charset + self.specials_last
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self.itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    def filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        eos_id, bos_id, pad_id = [self.stoi[s] for s in self.specials_first + self.specials_last]

        try:
            eos_idx = ids.index(eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]
        return probs, ids

    def decode(self, token_dists: Tensor, raw: bool = False):
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self.filter(probs, ids)
            tokens = self.ids2tok(ids)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

onnx_path = "tib-parseq-norbu+gbooks.onnx"
fname = '/Dataset/monlam-data/monlam.ai.ocr/OCR/training_images/I3CN47790527_6.tif'
img_transform = SceneTextDataModule.get_transform([32, 1024])
device = 'cpu'


ort_sess = ort.InferenceSession('tib-parseq-norbu+gbooks.onnx', providers=['CPUExecutionProvider'])
input_tensor = ort_sess.get_inputs()[0]

start_time = time.time()

gbooks_lines = open('../../data/gbooks_test_lines_for_tesseract.txt', 'r').readlines()
gbooks_lines = [line.strip() for line in gbooks_lines]
norbu_lines = open('../../data/norbu_test_lines_for_tesseract.txt', 'r').readlines()
norbu_lines = [line.strip() for line in norbu_lines]

paths = ['/Dataset/modern/training_lmdb/test/gbooks/', '/Dataset/modern/training_lmdb/test/norbu/']
charset = "ཡིད་གསུམཕརབཐོལའྱཤ །ྡྗེཆངཀཉནཔཟླཙ༑ཏཁྒྣྲ༌ྔྷྭཞྐྙཅྟྤྩཚ༈ཛཇྫྨྦཨཱཾཧཎ༄༅ཝྕ༔ཥྜྋཌ༼ཊ༴ྪཻཿ༽ༀྚཀྵ࿄	༎༉࿚ཽ྾༒ྺ༐࿙༜༆࿅ྀ྇ྰྠྃྵཪྴྛྞ༵༞ྻ༸྅༷ྶྥྑཋ༏྄༝࿐༹ྂ࿉༙༗ཬ"
output_ptr = open('onnx_output.txt', 'w')
total_imgs = 0

for path in paths:    
    lmdb_dataset = LmdbDataset(path, charset, max_label_len=450, min_image_dim=0, remove_whitespace=True, normalize_unicode=False, unlabelled=False, transform=img_transform)
    max_imgs = len(lmdb_dataset)

    for i in tqdm(range(max_imgs)):
        img, label, path = lmdb_dataset[i]
        imgname = path.decode().split('/')[-1].split('.')[0] 
        # print(label, path, imgname)
        if imgname in gbooks_lines or imgname in norbu_lines:
            img = img.unsqueeze(0).numpy()
            ort_outs = ort_sess.run(None, {input_tensor.name: img})
            logits = torch.from_numpy(ort_outs[0])
            outputs = logits.softmax(-1)

            token_decoder = TokenDecoder()
            pred, conf_scores = token_decoder.decode(outputs)
            if pred[] == label:
                output_ptr.write(f'{imgname}$$${label}$$${pred[0]}$$$correct\n')
            else:
                output_ptr.write(f'{imgname}$$${label}$$${pred[0]}$$$incorrect\n')
            
            total_imgs += 1
            
            
print('Total Images: ', total_imgs)
print(f'Avg Time Taken Per frame: {(time.time() - start_time) / total_imgs} seconds')