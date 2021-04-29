import torch


class AdaptiveLayer(torch.nn.Module):
    def __init__(self, size, adjustment='spectrum'):
        super(AdaptiveLayer, self).__init__()

        self.size = size
        self.adjustment = adjustment

        filter_size = self.size[:-1] + (self.size[-1] // 2 + 1,)
        self.register_parameter(name='frequency_filter',
                                param=torch.nn.Parameter(torch.empty(filter_size)))

        if self.adjustment in ('spectrum', 'spectrum_log'):
            torch.nn.init.ones_(self.frequency_filter)
        elif self.adjustment == 'phase':
            torch.nn.init.zeros_(self.frequency_filter)

    @staticmethod
    def spectrum_adjustment(rft_tensor, weights):
        return weights * rft_tensor

    @staticmethod
    def spectrum_log_adjustment(rft_tensor, weights):
        spectrum = torch.sqrt(torch.pow(rft_tensor.real, 2) + torch.pow(rft_tensor.imag, 2))
        return torch.where(spectrum == torch.tensor(0.0),
                           rft_tensor,
                           (rft_tensor / (spectrum + 1e-16)) * (torch.exp(weights * torch.log(1 + spectrum)) - 1))

    @staticmethod
    def phase_adjustment(rft_tensor, weights):
        return torch.complex(real=torch.cos(weights) * rft_tensor.real - torch.sin(weights) * rft_tensor.imag,
                             imag=torch.sin(weights) * rft_tensor.real + torch.cos(weights) * rft_tensor.imag)

    def forward(self, x):
        weights_size = (x.shape[0],) + tuple(1 for _ in range(len(self.size)))
        transformed_dimensions = tuple([dim for dim in range(1, len(weights_size))])

        rft_x = torch.fft.rfftn(x, dim=transformed_dimensions)
        if self.adjustment in ('spectrum', 'spectrum_log'):
            w = torch.nn.ReLU()(self.frequency_filter.repeat(weights_size).to(x.device))
            if self.adjustment == 'spectrum':
                adjusted_rft_x = self.spectrum_adjustment(rft_x, w)
            elif self.adjustment == 'spectrum_log':
                adjusted_rft_x = self.spectrum_log_adjustment(rft_x, w)
        elif self.adjustment == 'phase':
            w = torch.clamp(self.frequency_filter.repeat(weights_size).to(x.device),
                            min=0,
                            max=2 * torch.acos(torch.Tensor([-1])).item())
            adjusted_rft_x = self.phase_adjustment(rft_x, w)

        return torch.fft.irfftn(adjusted_rft_x, dim=transformed_dimensions)
