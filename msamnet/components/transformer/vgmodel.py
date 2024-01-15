import torch
from torch.autograd import Variable
import numpy
from seqtr.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils
from .one_stage import OneStageModel
# import torch.nn.functional as F
def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord
@MODELS.register_module()
class MSVGModel(OneStageModel):
    def __init__(self,
                 word_emb,
                 num_token,
                 vis_enc,
                 lan_enc,
                 head,
                 fusion):
        super(MSVGModel, self).__init__(
            word_emb,
            num_token,
            vis_enc,
            lan_enc,
            head,
            fusion)

    def forward_train(self,
                      img,
                      ref_expr_inds,
                      attention_mask,
                      img_metas,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      mass_center=None,
                      rescale=False):
        """Args:
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

        """
        x, y, y_mask = self.extract_visual_language(img, ref_expr_inds,attention_mask)

        x_mm = self.fusion(x)
        # x_mm = self.head.incorporate_coords(x_mm)
        # x, (y_words, y_mask) = self.extract_visual_language(img, ref_expr_inds)
        # x
        # shape [batchsize, 256, 80/72/64, 80])
        # shape [batchsize, 512, 40/36/32, 40])
        # shape [batchsize, 1024,20/18/16, 20])
        # y     [batchsize, 1,1024]

        # self.head.getprompt(x_mm) 
        # x_coord = generate_coord(x[1].shape[0],x[1].shape[2],x[1].shape[3])
        # x_mm = self.fusion(fvisu=x[1],fword=y_words,context_score = Variable(torch.ones(y_mask.shape[0],y_mask.shape[1]).cuda()),fcoord=x_coord,word_mask=y_mask)
        #3.2GB gpu usage
        # x_mm = torch.cat([x,])
 
        losses_dict, seq_out_dict = self.head.forward_train(
            x_mm, y, y_mask, img_metas,
            gt_bbox=gt_bbox,
            gt_mask_vertices=gt_mask_vertices)

        with torch.no_grad():
            predictions = self.get_predictions(
                seq_out_dict, img_metas, rescale=rescale)

        return losses_dict, predictions

    @torch.no_grad()
    def forward_test(self,
                     img,
                     ref_expr_inds,
                     attention_mask,
                     img_metas,
                     with_bbox=False,
                     with_mask=False,
                     rescale=False):
        """Args:
            img (tensor): [batch_size, c, h_batch, w_batch].

            ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rec/datasets/pipelines/formatting.py:CollectData`.

            with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
                which has slight differences.

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        """
        x, y,y_mask = self.extract_visual_language(img, ref_expr_inds,attention_mask)

        x_mm = self.fusion(x)
        # x_mm = self.head.incorporate_prompt(x_mm)
     
        # x_mm = self.fusion(fvisu=x[1], fword=y_words,context_score = Variable(torch.ones(y_mask.shape[0],y_mask.shape[1]).cuda()),fcoord=x_coord,word_mask=y_mask)
        # x_mm = self.head.incorporate_coords(x_mm)
        seq_out_dict = self.head.forward_test(
            x_mm,y, y_mask, img_metas,
            with_bbox=with_bbox,
            with_mask=with_mask)

        predictions = self.get_predictions(
            seq_out_dict, img_metas, rescale=rescale)

        return predictions

    def get_predictions(self, seq_out_dict, img_metas, rescale=False):
        """Args:
            seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

            rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
                back to `ori_shape`.
        """
        pred_bboxes, pred_masks = None, None
        with_bbox = 'seq_out_bbox' in seq_out_dict
        with_mask = 'seq_out_mask' in seq_out_dict

        if with_mask:
            seq_out_mask = seq_out_dict.pop('seq_out_mask')
            seq_out_mask = seq_out_mask.cpu().numpy()
            pred_masks = []
            for pred_mask, img_meta in zip(seq_out_mask, img_metas):
                h_pad, w_pad = img_meta['pad_shape'][:2]
                ends = numpy.argwhere(
                    pred_mask == self.head.end)[:, 0]
                if not len(ends) == 0:
                    pred_mask = pred_mask[:ends[0]]
                if len(pred_mask) % 2 != 0:  # must be even
                    pred_mask = pred_mask[:-1]

                scale_factor = [[w_pad, h_pad]
                                for _ in range(len(pred_mask) // 2)]
                scale_factor = numpy.array(scale_factor).reshape(-1)
                pred_mask = self.head.dequantize(pred_mask, scale_factor)
                pred_mask = pred_mask.astype(numpy.float64)

                if len(pred_mask) < 4:  # at least three points to make a mask
                    pred_mask = numpy.array(
                        [0, 0, 0, 0, 0, 0], order='F', dtype=numpy.uint8)
                    pred_mask = [pred_mask]
                elif len(pred_mask) == 4:
                    pred_mask = pred_mask[None]  # as a bbox
                else:
                    pred_mask = [pred_mask]  # as a polygon

                pred_rles = maskUtils.frPyObjects(
                    pred_mask, h_pad, w_pad)  # list[rle]
                pred_rle = maskUtils.merge(pred_rles)

                if rescale:
                    h_img, w_img = img_meta['ori_shape'][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(
                        pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict
                pred_masks.append(pred_rle)

        if with_bbox:
            seq_out_bbox = seq_out_dict.pop('seq_out_bbox')

            scale_factor = [img_meta['pad_shape'][:2][::-1]
                            for img_meta in img_metas]
            scale_factor = seq_out_bbox.new_tensor(scale_factor)
            scale_factor = torch.cat([scale_factor, scale_factor], dim=1)
            pred_bboxes = self.head.dequantize(seq_out_bbox, scale_factor)
            pred_bboxes = pred_bboxes.double()
            if rescale:  # from pad to ori
                scale_factors = [img_meta['scale_factor']
                                 for img_meta in img_metas]
                pred_bboxes /= pred_bboxes.new_tensor(scale_factors)

        return dict(pred_bboxes=pred_bboxes,
                    pred_masks=pred_masks)
