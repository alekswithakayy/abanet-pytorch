import torch
import torch.nn as nn

from types import MethodType

from .peak_backprop import pr_conv2d
from .peak_stimulation import PeakStimulation


class PeakResponseMapping(nn.Sequential):

    def __init__(self, args):
        super(PeakResponseMapping, self).__init__(args)
        if not args.return_peaks:
            raise ValueError('Model must return peaks to perform \
                peak response mapping')
        self.eval()
        self.peak_threshold = 30

    def forward(self, input):
        input.requires_grad_()
        logits, crms, peaks = super(PeakResponseMapping, self).forward(input)

        if crms.size(0) != 1:
            raise ValueError('Peak response mapping does not currently support \
                more than one image per batch')

        # Find index of class with highest confidence
        _, class_idx = torch.max(logits, dim=1)

        # Backpropagate peaks to get peak response maps
        prms = []
        valid_peaks = []
        grad_output = crms.new_empty(crms.size())
        for i in range(peaks.size(0)):
            peak = list(peaks[i])
            if peak[1] == class_idx:
                peak_val = crms[peak]
                if peak_val > self.peak_threshold:
                    grad_output.zero_()
                    grad_output[peak] = 1
                    if input.grad is not None: input.grad.zero_()
                    crms.backward(grad_output, retain_graph=True)
                    prm = input.grad.detach().sum(1).clone().clamp(min=0)
                    prms.append(prm / prm.sum())
                    valid_peaks.append(peaks[i])

        # return results
        crms = crms.detach()
        logits = logits.detach()

        if len(prms) == 0:
            return None
        else:
            valid_peaks = torch.stack(valid_peaks)
            prms = torch.cat(prms, 0)
            return logits, crms, valid_peaks, prms


    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)


    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward


    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        self._recover()
        return self


    def eval(self):
        super(PeakResponseMapping, self).eval()
        self._patch()
        return self
