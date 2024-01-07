import json
import numpy
from .utils import tokenize
from .builder import DATASETS
from .pipelines import Compose
from pydantic import ListMinLengthError
from torch.utils.data.dataset import Dataset
from seqtr.utils import get_root_logger, is_main

'''param:pipeline
    dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_bbox=True, dataset="RefCOCOUNC"),
    dict(type='LargeScaleJitter', out_max_size=640,
         jitter_min=0.3, jitter_max=1.4),
    # dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=['img', 'ref_expr_inds', 'gt_bbox']
]
'''
class BaseDataset(Dataset):
    def __init__(self,
                 imgsfile,
                 annsfile,
                 pipeline,
                 which_set='train',
                 img_source=['coco'],
                 word_emb_cfg=None):
        super(BaseDataset, self).__init__()
        assert isinstance(which_set, str) and which_set in [
            'train', 'val', 'testA', 'testB', 'test',
            'val_refcoco_unc', 'val_refcocoplus_unc', 'val_refcocog_umd',
            'val_flickr30k', 'val_referitgame_berkeley']
        self.which_set = which_set
        if len(img_source) == 1:
            assert img_source[0] in ['coco', 'visual-genome', 'flickr', 'saiaprtc12','rsvgd']# add 'rsvgd'
            self.imgsfile = imgsfile
        elif len(img_source) > 1:
            assert len(imgsfile) == len(img_source)
            assert isinstance(imgsfile, dict)
            self.imgsfile = imgsfile
        else:
            raise ListMinLengthError

        self.anns_all = json.load(open(annsfile, 'r'))

        self.token2idx, self.idx2token, self.word_emb = tokenize(annsfile,
                                                                 self.anns_all,
                                                                 word_emb_cfg)
        self.num_token = len(self.token2idx)

        if which_set == 'train':
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

    def _set_group_flag(self):
        self.flag = numpy.zeros(len(self), dtype=numpy.uint8)
        for i in range(len(self)):
            ann = self.anns_all[self.which_set][i]
            if ann['width'] / ann['height'] > 1:
                self.flag[i] = 1

    def __getitem__(self, index):
        results = {'ann': self.anns_all[self.which_set][index],
                   'which_set': self.which_set,
                   'token2idx': self.token2idx,# 整个字典，token化之后各phrase对应的idx
                   'imgsfile': self.imgsfile}# 图像dir：./data/images/mscoco/train2014
        '''
        {'bbox': [167.6, 256.14, 291.31, 144.66], 
        'image_id': 148343, 
        'height': 422, 
        'width': 480, 
        'category_id': 52, 
        'mask': [[204.52, 256.14, 188.55, 269.11, 194.54, 277.09, 174.59, 282.08, 167.6, 294.05, 169.6, 306.02, 185.56, 311.01, 188.55, 330.97, 212.5, 345.93, 239.43, 354.91, 282.33, 376.86, 310.26, 388.83, 322.24, 388.83, 329.22, 400.8, 335.21, 399.8, 335.21, 389.83, 354.16, 393.82, 378.1, 396.81, 380.1, 388.83, 413.02, 383.84, 431.98, 390.82, 441.95, 375.86, 443.95, 364.89, 437.96, 351.92, 434.97, 344.93, 438.96, 335.95, 443.95, 335.95, 458.91, 321.99, 442.95, 300.04, 325.23, 275.1, 260.38, 259.14]],
        'expressions': ['hot dog bottom one', 'closest dog', 'food nearest us']}
        '''
        '''image id 为148343的所有bbox和对应expressions
        [182.08, 174.49, 286.39, 116.64] ['top hot dog', 'top hot dog', 'top dog']
        [167.6, 256.14, 291.31, 144.66] ['hot dog bottom one', 'closest dog', 'food nearest us']
        [193.46, 0.0, 258.89, 193.46] ['jug', 'jar behind hot dogs', 'big white bottle of something top right']
        [19.96, 41.34, 177.73, 218.61] ['beverage', 'cup', 'cup on left']
        #
        f = open("instances.json",'r')
        content = f.read()
        b = json.loads(content)
        b = dict(b)
        for sample in b['train']:
    ...:     if sample['image_id'] == 148343:
    ...:         print(sample['bbox'],sample['expressions'])
        '''
        results = self.pipeline(results)
        #pipeline之后作为seqtr的输入，最后一个pipeline步骤为collectdata，参数keys=['img', 'ref_expr_inds', 'gt_bbox']
        '''{'img_metas': DataContainer({
             'filename': './data/images/mscoco/train2014/COCO_train2014_000000282471.jpg', 
             'expression': 'right', # 
             'ori_shape': (449, 640, 3), 
             'img_shape': (250, 357, 3), 
             'pad_shape': (256, 384, 3), 
             'scale_factor': array([0.5578125 , 0.55679287, 0.5578125 , 0.55679287]), 
             'img_norm_cfg': {
                'mean': array([0., 0., 0.], dtype=float32), 
                'std': array([1., 1., 1.], dtype=float32), 
                'to_rgb': True
              }
            }), 
            'img': DataContainer(tensor([[[121., 118., 116.,  ...,   0.,   0.,   0.],
            'ref_expr_inds': DataContainer(tensor([17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])), 
            'gt_bbox': DataContainer(tensor([189.9352,  52.8341, 352.9503, 249.0000], dtype=torch.float64))}
        '''
        # PIPELINE 最后的process是通过CollectData方法，取出keys制定的字段
        '''Args
            img (tensor): [batch_size, c, h_batch, w_batch].
            ref_expr_inds (tensor): [batch_size, max_token].

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `seqtr/datasets/pipelines/formatting.py:CollectData`.

            gt_bbox (list[tensor]): [4, ], in [tl_x, tl_y, br_x, br_y] format,
                the coordinates are in 'img_shape' scale.

            gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1, 
                the coordinates are in 'pad_shape' scale.

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        '''

        return results

    def __len__(self):
        return len(self.anns_all[self.which_set])


@DATASETS.register_module()
class RefCOCOUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOUNC-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOGoogle-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOgUMD(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOgUMD, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOg-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOgGoogle(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOgGoogle, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOg-{which_set} size: {len(self)}')


@DATASETS.register_module()
class RefCOCOPlusUNC(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RefCOCOPlusUNC, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RefCOCOPlusUNC-{which_set} size: {len(self)}')


@DATASETS.register_module()
class ReferItGameBerkeley(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(ReferItGameBerkeley, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'ReferItGameBerkeley-{which_set} size: {len(self)}')
# 注册数据类
@DATASETS.register_module()
class RSVG(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(RSVG, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'RSVG-{which_set} size: {len(self)}')

@DATASETS.register_module()
class Flickr30k(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(Flickr30k, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'Flick30k-{which_set} size: {len(self)}')


@DATASETS.register_module()
class Mixed(BaseDataset):
    def __init__(self, *args, **kwargs):
        which_set = kwargs['which_set']
        super(Mixed, self).__init__(*args, **kwargs)

        if is_main():
            logger = get_root_logger()
            logger.info(f'Mixed-{which_set} size: {len(self)}')
            logger.info(f'Mixed tokens: {len(self.token2idx)}')
