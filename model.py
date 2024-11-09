import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import CrossEntropyLoss

def get_inverse_sqrt_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases with inverse_sqrt
    from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to
    the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    decay_factor = num_warmup_steps ** 0.5 if num_warmup_steps > 0 else 1

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return decay_factor * float(current_step + 1) ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class BertCon(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, domain_number):
        """
        :param input_size: Size of the input features
        :param hidden_size: Size of the hidden layers in the MLP
        :param output_size: Size of the final shared encoder output
        :param domain_number: Number of classes for domain classification
        """
        super(BertCon, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )

        self.shared_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, output_size),
        )

        self.dom_loss1 = CrossEntropyLoss()
        self.dom_cls = nn.Linear(output_size, domain_number)
        self.tem = torch.tensor(0.05)

    def forward(self, input_embeddings, sent_labels=None, dom_labels=None, mode='train'):
        # Pass through MLP layers
        hidden = self.mlp(input_embeddings)
        batch_num = hidden.shape[0]
        h = self.shared_encoder(hidden)

        if mode == 'train':
            h = F.normalize(h, p=2, dim=1)
            sent_labels = sent_labels.unsqueeze(0).repeat(batch_num, 1).T
            rev_sent_labels = sent_labels.T
            rev_h = h.T
            similarity_mat = torch.exp(torch.matmul(h, rev_h) / self.tem)
            equal_mat = (sent_labels == rev_sent_labels).float()
            
            eye = torch.eye(batch_num)
            a = ((equal_mat - eye) * similarity_mat).sum(dim=-1) + 1e-5
            b = ((torch.ones(batch_num, batch_num) - eye) * similarity_mat).sum(dim=-1) + 1e-5

            loss = -(torch.log(a / b)).mean(-1)
            return loss

        elif mode == 'source':
            return F.normalize(h, p=2, dim=1)
