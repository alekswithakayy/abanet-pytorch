from types import MethodType

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.misc import imresize

from .peak_backprop import pr_conv2d
from .peak_stimulation import PeakStimulation


class PeakResponseMapping(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__(*args)

        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.peak_filter = kargs.get('filter_type', 'median')


    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)


    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward


    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        selected_instances = []
        while len(instance_list) > 0:
            instance = instance_list.pop(0)
            selected_instances.append(instance)
            src_mask = instance[2].astype(bool)
            src_peak_response = instance[3]
            def iou_filter(x):
                dst_mask = x[2].astype(bool)
                # IoU
                intersection = np.logical_and(src_mask, dst_mask).sum()
                union = np.logical_or(src_mask, dst_mask).sum()
                iou = intersection / (union + 1e-10)
                if iou < threshold:
                    return x
                else:
                    if merge_peak_response:
                        nonlocal src_peak_response
                        src_peak_response += x[3]
                    return None
            instance_list = list(filter(iou_filter, instance_list))
        return selected_instances


    def instance_seg(self, crms, peaks, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        crms = crms.squeeze().cpu().numpy()
        peaks = peaks.cpu().numpy()
        prms = peak_response_maps.cpu().numpy()

        img_height, img_width = prms.shape[1], prms.shape[2]

        # image size
        img_area = img_height * img_width

        # segment proposals off-the-shelf
        proposals = retrieval_cfg['proposals']

        # proposal contour width
        contour_width = retrieval_cfg.get('contour_width', 5)

        # limit range of proposal size
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))

        # selected number of proposals
        proposal_count = retrieval_cfg.get('proposal_count', 100)

        # nms threshold
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)

        # merge peak response during nms
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)

        # metric free parameters
        param = retrieval_cfg.get('param', None)

        # process each peak
        instance_list = []
        for i in range(len(prms)):
            class_idx = peaks[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(crms[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = prms[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                    (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                        peak_response_map[contour_mask].sum() - \
                        penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = self.instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold, merge_peak_response)
        return [dict(category=v[1], mask=v[2], prm=v[3]) for v in instance_list]


    def forward(self, input, class_threshold=0, peak_threshold=30, retrieval_cfg=None):
        assert input.dim() == 4, 'PeakResponseMapping only supports batch mode.'

        if self.inferencing:
            input.requires_grad_()

        # Feed-forward through network to compute class response maps (crms)
        crms = super(PeakResponseMapping, self).forward(input)

        if self.enable_peak_stimulation:
            if self.sub_pixel_locating_factor > 1:
                # Upsample class response maps to higher resolution
                crms = F.upsample(crms,
                    scale_factor=self.sub_pixel_locating_factor,
                    mode='bilinear', align_corners=True)
            # Aggregate responses from informative receptive fields
            peaks, aggregation = PeakStimulation.apply(crms, self.win_size,
                self.peak_filter)
        else:
            # Aggregate responses from all receptive fields
            peaks = None
            aggregation = F.adaptive_avg_pool2d(crms, 1)
            aggregation = aggregation.squeeze(2).squeeze(2)

        if self.inferencing:

            if not self.enable_peak_backprop:
                return aggregation, crms

            assert crms.size(0) == 1, 'Currently inference mode \
                with peak backpropagation only supports one image at a time.'

            if peaks is None:
                peaks, _ = PeakStimulation.apply(crms, self.win_size,
                    self.peak_filter)

            # Backpropagate peaks to get peak response maps (prms)
            prms = []
            valid_peaks = []
            grad_output = crms.new_empty(crms.size())
            for i in range(peaks.size(0)):
                peak = list(peaks[i])
                if aggregation[peak[0], peak[1]] >= class_threshold:
                    peak_val = crms[peak]
                    if peak_val > peak_threshold:
                        grad_output.zero_()
                        grad_output[peak] = 1
                        if input.grad is not None: input.grad.zero_()
                        crms.backward(grad_output, retain_graph=True)
                        prm = input.grad.detach().sum(1).clone().clamp(min=0)
                        prms.append(prm / prm.sum())
                        valid_peaks.append(peaks[i])

            # return results
            crms = crms.detach()
            aggregation = aggregation.detach()

            if len(prms) > 0:
                valid_peaks = torch.stack(valid_peaks)
                prms = torch.cat(prms, 0)
                if retrieval_cfg is None:
                    return aggregation, crms, valid_peaks, prms
                else:
                    return self.instance_seg(crms, valid_peaks, prms,
                                             retrieval_cfg)
            else:
                return None
        else:
            return aggregation


    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self


    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self
