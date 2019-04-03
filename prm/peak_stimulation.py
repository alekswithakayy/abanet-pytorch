import torch
import torch.nn.functional as F
from torch.autograd import Function


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, win_size, training):
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
        _, indices  = F.max_pool2d(input_padded, kernel_size=win_size,
                                   stride=1, return_indices=True)
        # Create boolean map of peak locations
        peak_map = (indices == index_map)

        batch_size, n_channels, h, w = input.size()
        f_input = input.view(batch_size, n_channels, h * w)

        # # During inference, filter peaks using standard deviation
        # if not training:
        #     # Use mean and std to find appropriate threshold
        #     mean = torch.mean(f_input, dim=2)
        #     mean = mean.contiguous().view(batch_size, n_channels, 1, 1)
        #     std = torch.std(f_input, dim=2)
        #     std = std.contiguous().view(batch_size, n_channels, 1, 1)
        #     # Filter out peaks less than n standard deviations
        #     # from mean peak response during inference
        #     n_std = 1.5
        #     thresh = mean + std * n_std
        #     valid_peaks = (input >= thresh)
        #
        #     # Use max to ensure at least one peak found (avoid div by 0 later)
        #     max, _ = torch.max(f_input, dim=2)
        #     max = max.contiguous().view(batch_size, n_channels, 1, 1)
        #     max_peak = (input == max)
        #
        #     # Filter peak map
        #     peak_map = (peak_map & (valid_peaks | max_peak))
        #
        # # During training, filter peaks lower than kth highest peak response
        # else:

        # Filters out peaks not belonging to
        # top percent of peak responses
        top_x_percent = 0.5 if training else 0.5
        k = int(h * w * top_x_percent)
        topk, _ = torch.topk(f_input, k, dim=2)

        # Use kth highest peak response as threshold
        thresh = topk[:, :, -1]
        thresh = thresh.contiguous().view(batch_size, n_channels, 1, 1)
        valid_peaks = (input >= thresh)

        # Filter peak map
        peak_map = (peak_map & valid_peaks)

        # Save peak response map for backprop
        peak_map = peak_map.half()
        ctx.save_for_backward(input, peak_map)

        # Create list of peak locations (indices)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        # Calculate average of all peaks
        logits = (input * peak_map).view(batch_size, n_channels, -1).sum(2)
        n_peaks = peak_map.view(batch_size, n_channels, -1).sum(2)
        logits = logits / n_peaks

        return peak_list, logits


    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, n_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, n_channels, 1, 1)
        return (grad_input,) + (None,) * ctx.num_flags
