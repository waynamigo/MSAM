dataset = 'RSVGD'
data_root = './data/'
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.])

train_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=20, with_bbox=True, dataset="RSVG"),
    dict(type='LargeScaleJitter', out_max_size=640,
         jitter_min=0.3, jitter_max=1.4),
    # dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]
val_pipeline = [
    dict(type='LoadImageAnnotationsFromFile',
         max_token=20, with_bbox=True, dataset="RSVG"),
    dict(type='Resize', img_scale=(640, 640)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='CollectData', keys=[
         'img', 'ref_expr_inds', 'gt_bbox'])
]

test_pipeline = val_pipeline.copy()
# only for lstm. BERT is applied with its own dictionary in ~/data/weights/.
# word_emb_cfg = dict(type='GloVe')

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        which_set='train',
        img_source=['rsvgd'],
        annsfile=data_root + 'annotations/rsvgd/instances.json',
        imgsfile=data_root + 'images/rsvgd',
        pipeline=train_pipeline,
        word_emb_cfg=word_emb_cfg),
    val=dict(
        type=dataset,
        which_set='val',
        img_source=['rsvgd'],
        annsfile=data_root + 'annotations/rsvgd/instances.json',
        imgsfile=data_root + 'images/rsvgd',
        pipeline=val_pipeline,
        word_emb_cfg=word_emb_cfg),
    test=dict(
        type=dataset,
        which_set='test',
        img_source=['rsvgd'],
        annsfile=data_root + 'annotations/rsvgd/instances.json',
        imgsfile=data_root + 'images/rsvgd',
        pipeline=test_pipeline,
        word_emb_cfg=word_emb_cfg),
)
