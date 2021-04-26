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

    def spectrum_adjustment(self, rfft_tensor, weights):
        init_spectrum = torch.sqrt(torch.pow(rfft_tensor.real, 2) + torch.pow(rfft_tensor.imag, 2))
        if self.adjustment == 'spectrum':
            return torch.where(init_spectrum == torch.tensor(0.0), rfft_tensor, weights * rfft_tensor)
        elif self.adjustment == 'spectrum_log':
            return torch.where(init_spectrum == torch.tensor(0.0),
                               rfft_tensor,
                               (rfft_tensor / (init_spectrum + 1e-16)) * (torch.exp(weights * torch.log(1 + init_spectrum)) - 1))

    def phase_rotation(self, rfft_tensor, weights):
        return torch.complex(real=torch.cos(weights) * rfft_tensor.real - torch.sin(weights) * rfft_tensor.imag,
                             imag=torch.sin(weights) * rfft_tensor.real + torch.cos(weights) * rfft_tensor.imag)

    def phase_adjustment(self, rfft_tensor, weights):
        init_spectrum = torch.pow(rfft_tensor.real, 2) + torch.pow(rfft_tensor.imag, 2)
        return torch.where(init_spectrum == torch.tensor(0.0), rfft_tensor, self.phase_rotation(rfft_tensor, weights))

    def forward(self, x):
        weights_size = (x.shape[0],) + tuple(1 for _ in range(len(self.size)))
        transformed_dimensions = tuple([dim for dim in range(1, len(weights_size))])
        rfft_x = torch.fft.rfftn(x, dim=transformed_dimensions)

        if self.adjustment in ('spectrum', 'spectrum_log'):
            w = torch.nn.ReLU()(self.frequency_filter.repeat(weights_size).to(x.device))
            return torch.fft.irfftn(self.spectrum_adjustment(rfft_x, w), dim=transformed_dimensions)
        elif self.adjustment == 'phase':
            w = torch.clamp(self.frequency_filter.repeat(weights_size).to(x.device),
                            min=0,
                            max=2 * torch.acos(torch.Tensor([-1])).item())
            return torch.fft.irfftn(self.phase_adjustment(rfft_x, w), dim=transformed_dimensions)
