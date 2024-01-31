import sennet.custom_modules.losses.custom_losses as custom_losses
import torch.nn.functional as F
import torch


LOSSES_WITH_BCE_INPUT = (custom_losses.BCELoss, custom_losses.FocalLoss, custom_losses.TopKPercentBCELoss)


class CombinedLoss(torch.nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.loss_functions = []
        for loss_info in config['loss']:
            loss_type = loss_info['type']
            weight = loss_info.get('weight', 1.0)
            kwargs = loss_info.get('kwargs', {})
            loss_instance = getattr(custom_losses, loss_type)(**kwargs)
            self.loss_functions.append((loss_instance, weight))

    def __call__(self, input, target, weights):
        total_loss = 0.0
        sum_weights = 0.0

        # Precompute BCE loss if necessary
        bce_loss = None
        if any(isinstance(loss_fn, LOSSES_WITH_BCE_INPUT) for loss_fn, _ in self.loss_functions):
            bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")

        if weights is not None:
            bce_loss = bce_loss * weights

        for loss_fn, weight in self.loss_functions:
            if weight < 1e-3:
                continue
            if isinstance(loss_fn, LOSSES_WITH_BCE_INPUT):
                loss_value = loss_fn(bce_loss)
            else:
                loss_value = loss_fn(input, target)
            
            total_loss += weight * loss_value
            sum_weights += weight

        total_loss = total_loss / sum_weights
        
        return total_loss
