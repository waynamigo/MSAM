import argparse
import os.path as osp
from typing import Sequence
from mmcv import Config, DictAction
from mmcv.utils import mkdir_or_exist
from seqtr.apis import inference_model


def parse_args():
    parser = argparse.ArgumentParser(description="macvg-inference")
    parser.add_argument('config', help='inference config file path.')
    parser.add_argument(
        'checkpoint', help='the checkpoint file to load from.')
    parser.add_argument(
        '--output-dir', help='directory where inference results will be saved.')
    parser.add_argument('--with-gt', action='store_true',
                        help='draw ground-truth bbox/mask on image if true.')
    parser.add_argument('--no-overlay', action='store_false', dest='overlay')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--which-set',
        type=str,
        nargs='+',
        default='val',
        help="evaluation which_sets, which depends on the dataset, e.g., \
        'val', 'testA', 'testB' for RefCOCO(Plus)UNC, and 'val', 'test' for RefCOCOgUMD.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.checkpoint = args.checkpoint
    assert args.which_set is not None, "please specify at least one which_set to inference on."
    if isinstance(args.which_set, str):
        cfg.which_set = [args.which_set]
    elif isinstance(args.which_set, Sequence):
        cfg.which_set = args.which_set
    cfg.overlay = args.overlay
    cfg.output_dir = args.output_dir
    cfg.with_gt = args.with_gt
    cfg.rank = 0
    cfg.distributed = False

    for which_set in cfg.which_set:
        mkdir_or_exist(
            osp.join(args.output_dir, cfg.dataset + "_" + which_set))

    inference_model(cfg)
import mmcv
import torch
import os.path as osp

from seqtr.utils import load_checkpoint, get_root_logger
from seqtr.core import imshow_expr_bbox, imshow_expr_mask
from seqtr.models import build_model, ExponentialMovingAverage
from seqtr.datasets import extract_data, build_dataset, build_dataloader

try:
    import apex
except:
    pass


def inference_model(cfg):
    datasets_cfg = [cfg.data.train]
    for which_set in cfg.which_set:
        datasets_cfg.append(eval(f"cfg.data.{which_set}"))

    datasets = list(map(build_dataset, datasets_cfg))
    dataloaders = list(
        map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model,
                        word_emb=datasets[0].word_emb,
                        num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(
            model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.ema:
        model_ema = ExponentialMovingAverage(
            model, cfg.ema_factor)
    else:
        model_ema = None
    load_checkpoint(model, model_ema, None, cfg.checkpoint)
    if cfg.ema:
        model_ema.apply_shadow()

    model.eval()
    logger = get_root_logger()
    with_bbox, with_mask = False, False
    for i, which_set in enumerate(cfg.which_set):
        logger.info(f"inferencing on split {which_set}")
        prog_bar = mmcv.ProgressBar(len(datasets[i+1]))
        with torch.no_grad():
            for batch, inputs in enumerate(dataloaders[i+1]):
                gt_bbox, gt_mask, is_crowd = None, None, None
                if 'gt_bbox' in inputs:
                    with_bbox = True
                    gt_bbox = inputs.pop('gt_bbox').data[0]
                if 'gt_mask_rle' in inputs:
                    with_mask = True
                    gt_mask = inputs.pop('gt_mask_rle').data[0]
                if 'is_crowd' in inputs:
                    inputs.pop('is_crowd').data[0]

                if not cfg.distributed:
                    inputs = extract_data(inputs)

                img_metas = inputs['img_metas']
                batch_size = len(img_metas)

                predictions = model(**inputs,
                                    return_loss=False,
                                    rescale=True,
                                    with_bbox=with_bbox,
                                    with_mask=with_mask)

                pred_bboxes = [None for _ in range(batch_size)]
                if with_bbox:
                    pred_bboxes = predictions.pop('pred_bboxes')
                pred_masks = [None for _ in range(batch_size)]
                if with_mask:
                    pred_masks = predictions.pop('pred_masks')

                for j, (img_meta, pred_bbox, pred_mask) in enumerate(zip(img_metas, pred_bboxes, pred_masks)):
                    filename, expression = img_meta['filename'], img_meta['expression']
                    bbox_gt, mask_gt = None, None
                    if cfg.with_gt and with_bbox:
                        bbox_gt = gt_bbox[j]
                    if cfg.with_gt and with_mask:
                        mask_gt = gt_mask[j]

                    outfile = osp.join(
                        cfg.output_dir,
                        cfg.dataset + "_" + which_set,
                        expression.replace(" ", "_") + "_" + osp.basename(filename))

                    if with_bbox:
                        imshow_expr_bbox(filename,
                                         pred_bbox,
                                         outfile,
                                         gt_bbox=bbox_gt)
                    if with_mask:
                        imshow_expr_mask(filename,
                                         pred_mask,
                                         outfile,
                                         gt_mask=mask_gt,
                                         overlay=cfg.overlay)

                    prog_bar.update()
    if cfg.ema:
        model_ema.restore()