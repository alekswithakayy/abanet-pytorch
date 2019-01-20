import torch
import torch.nn.functional as F
from torch.autograd import Function


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, win_size, peak_filter):
        ctx.num_flags = 4

        assert win_size % 2 == 1, 'Window size for peak finding must be odd.'

        # Pad input
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        input_padded = padding(input)

        # Create tensor with index values as tensor elements
        batch_size, n_channels, h, w = input_padded.size()
        index_map = torch.arange(0, h * w).long().view(1, 1, h, w)
        index_map = index_map[:, :, offset: -offset, offset: -offset]
        index_map = index_map.to(input.device)

        # Find index values of peaks
        _, indices  = F.max_pool2d(input_padded, kernel_size=win_size, stride=1,
                                   return_indices=True)

        # Create boolean map of peak locations
        peak_map = (indices == index_map)

        # Filter peaks
        if peak_filter:
            batch_size, n_channels, h, w = input.size()
            filter_input = input.view(batch_size, n_channels, h * w)
            if peak_filter == 'median':
                thresh, _ = torch.median(filter_input, dim=2)
            elif peak_filter == 'mean':
                thresh = torch.mean(filter_input, dim=2)
            elif peak_filter == 'max':
                thresh, _ = torch.max(filter_input, dim=2)
            std = torch.std(filter_input, dim=2)
            std = std.contiguous().view(batch_size, n_channels, 1, 1)
            thresh = thresh.contiguous().view(batch_size, n_channels, 1, 1)
            thresh = (thresh + std * 0.75)
            peak_mask = input >= thresh
            peak_map = (peak_map & peak_mask)

        peak_map = peak_map.float()
        ctx.save_for_backward(input, peak_map)

        # Create list of peak locations (indices)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        # Calculate average of all peaks
        aggregation = (input * peak_map).view(batch_size, n_channels, -1).sum(2)
        n_peaks = peak_map.view(batch_size, n_channels, -1).sum(2)
        peak_average = aggregation / n_peaks

        return peak_list, peak_average


    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, n_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, n_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags
