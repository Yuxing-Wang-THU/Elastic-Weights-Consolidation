import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.autograd as autograd
import utils
from torch.nn import functional as F

USE_CUDA = torch.cuda.is_available()

def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var


class EWC(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ewc_lambda = args.lamda
        self.tasks_encountered = []

        self.fisher = {}
        self.optpar = {}

    def forward(self, named_params):
        net_loss = Variable(torch.Tensor([0]))
        if not self.ewc_lambda:
            return net_loss
        for task_id in self.tasks_encountered:
            for name, param in named_params:
                fisher = Variable(self.fisher[task_id][name])
                optpar = Variable(self.optpar[task_id][name])
                net_loss += (fisher * (optpar - param).pow(2)).sum() 
        return net_loss * self.ewc_lambda/2

    def regularize(self, named_params):
        """Calculate the EWC regularization component in the overall loss.
        For all the tasks encountered in past, L2-norm loss is calculated
        between current model parameters and optimal parameters of previous
        tasks, weighted by terms from fisher matrix.

        Arguments
        =========
        named_params : generator
            Named parameters of model to be regularized.
        """
        return self.forward(named_params)
    
    # Update the Fisher Information
    def update_fisher_optpar(self, model, current_itr, dataset, sample_size, batch_size=32, consolidate=True):
        if consolidate:
            if self.args.online:
                current_itr = 1
                self.tasks_encountered = [1]
            else:
                self.tasks_encountered.append(current_itr)
            
        data_loader = utils.get_data_loader(dataset, batch_size)
        losses = []
        for x, y in data_loader:
            x = x.view(batch_size, -1)
            x = Variable(x).cuda() if USE_CUDA else Variable(x)
            y = Variable(y).cuda() if USE_CUDA else Variable(y)

            losses.append(
                F.log_softmax(model(x), dim=1)[range(batch_size), y.data]
            )
            if len(losses) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        sample_losses = torch.cat(losses).unbind()
        sample_grads = zip(*[autograd.grad(l, model.parameters(), retain_graph=(i < len(sample_losses))) 
        for i, l in enumerate(sample_losses, 1)])

        sample_grads = [torch.stack(gs) for gs in sample_grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in sample_grads]
        self.fisher[current_itr] = {}
        self.optpar[current_itr] = {}

        for (name, param), fisher in zip(model.named_parameters(), fisher_diagonals):
            self.optpar[current_itr][name] = param.data.clone()
            self.fisher[current_itr][name] = fisher.detach()