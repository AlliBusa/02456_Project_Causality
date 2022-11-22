from collections import defaultdict
import torch
from torch import nn, Tensor
from typing import Tuple, Dict
import numpy as np

class VariationalInference(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.a = 0

    def forward(self, model: nn.Module, x: Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x)

        # unpack outputs
        # Get parameters of the prior and posterior and px and z's
        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]

        # evaluate log probabilities
        # Whats the probability of getting x and z given the estimated distributions
        log_px = reduce(px.log_prob(x))
        # pdb.set_trace()
        self.a += 1
        # print(self.a)
        # pdb.set_trace()
        # print(qz.mu)
        # print(qz.mu.shape)
        log_pz = reduce(pz.log_prob(z))
        log_qz = reduce(qz.log_prob(z))

        # compute the ELBO with and without the beta parameter:
        # `L^beta = E_q [ log p(x|z) ] - beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl = log_qz - log_pz
        elbo = log_px - kl  # <- your code here
        beta_elbo = log_px - self.beta * kl  # <- your code here

        # loss
        loss = -beta_elbo.mean()

        # prepare the output
        with torch.no_grad():
            diagnostics = {"elbo": elbo, "log_px": log_px, "kl": kl}

        return loss, diagnostics, outputs


####################################################################

## Training Loop

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)
    
def run_training(vae, train_loader, test_loader, optimizer, vi, num_epochs=50):

    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    epoch = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")

    # move the model to the device
    vae = vae.to(device)

    # training..
    while epoch < num_epochs:
        epoch += 1
        training_epoch_data = defaultdict(list)
        vae.train()

        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        # for x, y in train_loader:
        for x in train_loader:

            x = x.to(device)

            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x)

            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()

            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]

        # gather data for the full epoch
        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        # Evaluate on a single batch, do not propagate gradients
        with torch.no_grad():
            vae.eval()

            # Just load a single batch from the test loader
            # x, y = next(iter(test_loader))
            x = next(iter(test_loader))
            x = x.to(device)

            # perform a forward pass through the model and compute the ELBOwhy
            loss, diagnostics, outputs = vi(vae, x)

            # gather data for the validation step
            for k, v in diagnostics.items():
                validation_data[k] += [v.mean().item()]

        # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
        # make_vae_plots(vae, x, y, outputs, training_data, validation_data)

        return outputs, loss, diagnostics
