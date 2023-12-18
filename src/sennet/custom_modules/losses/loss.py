import sennet.custom_modules.losses.custom_losses as custom_losses
import torch.nn.functional as F

class CombinedLoss:
    def __init__(self, config):
        self.loss_functions = []
        for loss_info in config['loss']:
            loss_type = loss_info['type']
            weight = loss_info.get('weight', 1.0)
            kwargs = loss_info.get('kwargs', {})
            loss_instance = getattr(custom_losses, loss_type)(**kwargs)
            self.loss_functions.append((loss_instance, weight))

    def __call__(self, input, target):
        total_loss = 0.0
        sum_weights = 0.0

        # Precompute BCE loss if necessary
        bce_loss = None
        if any(isinstance(loss_fn, (custom_losses.BCELoss, custom_losses.FocalLoss)) for loss_fn, _ in self.loss_functions):
            bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        for loss_fn, weight in self.loss_functions:
            if isinstance(loss_fn, (custom_losses.BCELoss, custom_losses.FocalLoss)):
                loss_value = loss_fn(bce_loss)
            else:
                loss_value = loss_fn(input, target)
            
            total_loss += weight * loss_value
            sum_weights += weight

        return total_loss / sum_weights
