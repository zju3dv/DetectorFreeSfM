import torch
import torch.nn as nn

class DetectorWrapper(nn.Module):
    def __init__(self,
                 detector=None,
                 detector_type="OnGrid",
                 fullcfg=None):
        super().__init__()
        assert detector_type in ['OnGrid', 'SuperPoint', 'SuperPointEC', 'SIFT'] \
            or 'and grid' in detector_type
        self.detector_type = detector_type

        if detector_type == 'OnGrid':
            assert detector is None
            self.detector = None

    @torch.no_grad()
    def forward(self, batch):
        if self.detector_type == 'OnGrid':
            pass
        else:
            raise NotImplementedError