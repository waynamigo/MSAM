{'img_metas': DataContainer({
             'filename': './data/images/mscoco/train2014/COCO_train2014_000000282471.jpg', 
             'expression': 'right', 
             'ori_shape': (449, 640, 3), 
             'img_shape': (250, 357, 3), 
             'pad_shape': (256, 384, 3), 
             'scale_factor': array([0.5578125 , 0.55679287, 0.5578125 , 0.55679287]), 
             'img_norm_cfg': {
                'mean': array([0., 0., 0.], dtype=float32), 
                'std': array([1., 1., 1.], dtype=float32), 
                'to_rgb': True}
            })

inputs from dataloader{
    input keys ['img_metas', 'img', 'ref_expr_inds', 'gt_bbox']
}

inputs = extract_data(inputs)

//inputs extrace_data 后得到的数据作为输入
inputs after extraction{
    input keys ['img_metas', 'img', 'ref_expr_inds', 'gt_bbox']
}
losses, predictions = model(**inputs, rescale=False)



#DEFINE 
model = dict(
    type='SeqTR',
    vis_enc=dict(
        type='DarkNet53',
        pretrained='./data/weights/darknet.weights',
        freeze_layer=2,
        out_layer=(6, 8, 13)),
    lan_enc=dict(
        type='LSTM',
        lstm_cfg=dict(
            type='gru',
            num_layers=1,
            hidden_size=512,
            dropout=0.0,
            bias=True,
            bidirectional=True,
            batch_first=True),
        freeze_emb=True,
        output_cfg=dict(type='max')),
    fusion=dict(
        type='SimpleFusion', vis_chs=[256, 512, 1024], direction='bottom_up'),
    head=dict(
        type='SeqHead',
        in_ch=1024,
        num_bin=1000,
        multi_task=False,
        shuffle_fraction=-1,
        mapping='relative',
        top_p=-1,
        num_ray=-1,
        det_coord=[0],
        det_coord_weight=1.5,
        loss=dict(type='LabelSmoothCrossEntropyLoss', neg_factor=0.1),
        predictor=dict(
            num_fcs=3,
            in_chs=[256, 256, 256],
            out_chs=[256, 256, 1001],
            fc=[
                dict(
                    linear=dict(type='Linear', bias=True),
                    act=dict(type='ReLU', inplace=True),
                    drop=None),
                dict(
                    linear=dict(type='Linear', bias=True),
                    act=dict(type='ReLU', inplace=True),
                    drop=None),
                dict(
                    linear=dict(type='Linear', bias=True), act=None, drop=None)
            ]),
        transformer=dict(
            type='AutoRegressiveTransformer',
            encoder=dict(
                num_layers=6,
                layer=dict(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True)),
            decoder=dict(
                num_layers=3,
                layer=dict(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.1,
                    activation='relu',
                    batch_first=True))),
        x_positional_encoding=dict(
            type='SinePositionalEncoding2D', num_feature=128, normalize=True),
        seq_positional_encoding=dict(
            type='LearnedPositionalEncoding1D',
            num_embedding=5,
            num_feature=256)))

#IMG_metas
[{'filename': './data/images/mscoco/train2014/COCO_train2014_000000492383.jpg', 
'expression': 'banana in middle just below one with brown spots', 
'ori_shape': (480, 640, 3), 
'img_shape': (480, 640, 3), 
'pad_shape': (480, 640, 3), 
'scale_factor': array([1., 1., 1., 1.]), 
'img_norm_cfg': {
    'mean': array([0., 0., 0.], dtype=float32), 
    'std': array([1., 1., 1.], dtype=float32), 
    'to_rgb': True}, 
    'batch_input_shape': (544, 640)}, 
{'filename': './data/images/mscoco/train2014/COCO_train2014_000000431112.jpg',
 'expression': 'wine glass on right', 'ori_shape': (427, 640, 3), 'img_shape': (427, 640, 3), 'pad_shape': (448, 640, 3), 
 'scale_factor': array([1., 1., 1., 1.]), 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 
 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (544, 640)}, 
 
 {'filename': './data/images/mscoco/train2014/COCO_train2014_000000524131.jpg',
  'expression': 'center horse', 
  'ori_shape': (467, 640, 3),
   'img_shape': (290, 397, 3), 
   'pad_shape': (320, 416, 3),
    'scale_factor': array([0.6203125 , 0.62098501, 0.6203125 , 0.62098501]), 
    'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 
    'batch_input_shape': (544, 640)}, {'filename': './data/images/mscoco/train2014/COCO_train2014_000000321980.jpg', 
    'expression': 'table top to left of red notebook', 'ori_shape': (447, 640, 3), 'img_shape': (447, 640, 3), 'pad_shape': 
    (448, 640, 3), 'scale_factor': array([1., 1., 1., 1.]), 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([1., 1., 1.], 
    dtype=float32), 'to_rgb': True}, 'batch_input_shape': (544, 640)}, 
    {'filename': './data/images/mscoco/train2014/COCO_train2014_000000267049.jpg', 'expression': 'man in clock', 'ori_shape': (480, 640, 3),
    'img_shape': (291, 388, 3), 'pad_shape': (320, 416, 3), 'scale_factor': array([0.60625, 0.60625, 0.60625, 0.60625]), '
    img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 
    'batch_input_shape': (544, 640)}, {'filename': './data/images/mscoco/train2014/COCO_train2014_000000569742.jpg',
     'expression': 'man on far left', 'ori_shape': (427, 640, 3), 'img_shape': (427, 640, 3), 'pad_shape': (448, 640, 3), 
     'scale_factor': array([1., 1., 1., 1.]), 'img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 
     'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (544, 640)},
      {'filename': './data/images/mscoco/train2014/COCO_train2014_000000261202.jpg', 'expression': 'right rabbit', 'ori_shape': (480, 640, 3), 
      'img_shape': (362, 483, 3), 'pad_shape': (384, 512, 3), 'scale_factor': array([0.7546875 , 0.75416667, 0.7546875 , 0.75416667]), '
      img_norm_cfg': {'mean': array([0., 0., 0.], dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 
      'batch_input_shape': (544, 640)}, 
      {'filename': './data/images/mscoco/train2014/COCO_train2014_000000352073.jpg', 'expression': 'girl on right', 'ori_shape': (427, 640, 3), 
      'img_shape': (427, 640, 3), 'pad_shape': (448, 640, 3), 'scale_factor': array([1., 1., 1., 1.]), 'img_norm_cfg': {'mean': array([0., 0., 0.],
       dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': True}, 'batch_input_shape': (544, 640)}]



(0): ConvModule(
    (conv): Conv2d(1792, 1792, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activate): LeakyReLU(negative_slope=0.1, inplace=True)
)
(1): ConvModule(
    (conv): Conv2d(1792, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activate): LeakyReLU(negative_slope=0.1, inplace=True)
)
