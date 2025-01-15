import torch


def get_CUDA_memory_allocation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('No CUDA found')

    else:
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(f"Total memory(MB): {total} /"
              f"Memory used (MB): {mem_used_MB} /"
              f"Memory free (MB): {free}")


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Averager:      ##Return the average loss
    def __init__(self):
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0

    def send(self, new_losses: dict):
        for key, value in self.current_losses.items():
            self.current_losses[key] = new_losses[key] + value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return {
                "regression": 0.0,
                "classification": 0.0,
                "total": 0.0
            }
        else:
            dict_to_return = {}
            for key, value in self.current_losses.items():
                dict_to_return[key] = value / self.iterations

            return dict_to_return

    def reset(self):
        self.current_losses = {
            "regression": 0.0,
            "classification": 0.0,
            "total": 0.0
        }
        self.iterations = 0.0