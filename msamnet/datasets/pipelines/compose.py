import collections
from ..builder import PIPELINES
from mmcv.utils import build_from_cfg
# mmcv/mmcv/utils/registry.py
# build_from_cfg
''' dict(type='LoadImageAnnotationsFromFile',
         max_token=15, with_bbox=True, dataset="RefCOCOUNC"),
    dict(type='LargeScaleJitter', out_max_size=640,
         jitter_min=0.3, jitter_max=1.4),
    # dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=['img', 'ref_expr_inds', 'gt_bbox']'''
@PIPELINES.register_module()
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)# transform(data) 一步一步得到最终transform后的result
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None
        return results
