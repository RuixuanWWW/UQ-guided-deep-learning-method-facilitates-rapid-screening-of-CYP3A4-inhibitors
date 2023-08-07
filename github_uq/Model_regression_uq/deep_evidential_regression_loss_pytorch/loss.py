# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements the evidential loss using Normal Inverse Gamma Distribution
Use this function when you want to model your regression output as a 
normal inverse gamma distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class EvidentialLossNLL(nn.Module):
  """The evidential loss function on a matrix.

  This class is implemented with slight modifications from the paper. The major
  change is in the regularizer parameter mentioned in the paper. The regularizer
  mentioned in the paper didnot give the required results, so we modified it 
  with the KL divergence regularizer from the paper. In orderto overcome the problem
  that KL divergence are missing near zero so we add the minimum values to alpha,
  beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)

  This class only allows for rank-4 inputs for the output `targets`, and expectes
  `inputs` be of the form [mu, alpha, beta, lambda] 

  alpha, beta and lambda needs to be positive values.
  """

  def __init__(self, debug=False, return_all=False):
    """Sets up loss function.

    Args:
      debug: When set to 'true' prints all the intermittent values
      return_all: When set to 'true' returns all loss values without taking average

    """
    super(EvidentialLossNLL, self).__init__()

    self.debug = debug
    self.return_all_values = return_all
    self.MAX_CLAMP_VALUE = 5.0   # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

  def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
    alpha_2 = torch.ones_like(mu1)*1.0
    beta_2 = torch.ones_like(mu1)*0.1
    lambda_2 = torch.ones_like(mu1)*1.0

    t1 = 0.5 * (alpha_1/beta_1) * ((mu1 - mu2)**2)  * lambda_2
    #t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
    t2 = 0.5*lambda_2/lambda_1
    t3 = alpha_2*torch.log(beta_1/beta_2)
    t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
    t5 = (alpha_1-alpha_2)*torch.digamma(alpha_1)
    t6 = -(beta_1 - beta_2)*(alpha_1/beta_1)
    return (t1+t2-0.5+t3+t4+t5+t6)

  def forward(self, inputs, targets,lam, epsilon):
    """ Implements the loss function 

    Args:
      inputs: The output of the neural network. inputs has 4 dimension 
        in the format [mu, alpha, beta, lambda]. Must be a tensor of
        floats
      targets: The expected output

    Returns:
      Based on the `return_all` it will return mean loss of batch or individual loss

    """
    assert torch.is_tensor(inputs)
    assert torch.is_tensor(targets)
#     assert (inputs[:,1] > 0).all()
#     assert (inputs[:,2] > 0).all()
#     assert (inputs[:,3] > 0).all()

    targets = targets.view(-1)
    mu = inputs[:,0].view(-1) #first column is mu,predicted value
    loglambdas = inputs[:,1].view(-1) #lamda
    logalphas = inputs[:,2].view(-1) #alpha to limit >1
    logbetas = inputs[:,3].view(-1) #beta
    
    if self.debug:
      print("a :", a)
      print("b :", b)
      print("l :", l)
    
    # new_loss_230618
    
    min_val = 1e-6
    
    v = torch.nn.Softplus()(loglambdas) + min_val
    alpha = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
    beta = torch.nn.Softplus()(logbetas) + min_val
    
    
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v) \
        - alpha*torch.log(twoBlambda) \
        + (alpha+0.5) * torch.log(v*(targets-mu)**2 + twoBlambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha+0.5)

    L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg #torch.mean(reg, dim=-1)

#     Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
#     print('lambda=',lam)
    loss = L_NLL + lam * (L_REG - epsilon)
    ret_loss = loss.mean()
    
    ######initial_loss
    
#     J1 = torch.lgamma(a - 0.5) 
#     J2 = -torch.log(torch.tensor([4.0]))

#     device = ('cuda' if torch.cuda.is_available() else 'cpu')

#     J2=J2.to(device)
#     J3 = -torch.lgamma(a) 
#     J4 = -torch.log(l) 
#     J5 = -0.5*torch.log(b) 
#     J6 = torch.log(2*b*(1 + l) + (2*a - 1)*l*(y-targets)**2)
      
#     if self.debug:
#       print("lgama(a - 0.5) :", J1)
#       print("log(4):", J2)
#       print("lgama(a) :", J3)
#       print("log(l) :", J4)
#       print("log( ---- ) :", J6)

#     J = J1 + J3 + J4 + J5 + J6 + J2

#     Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)
    
#     if self.debug:
#       print ("KL ",Kl_divergence.data.numpy())
#     loss = torch.exp(J) + Kl_divergence

#     if self.debug:
#       print ("loss :", loss.mean())
    

#     if self.return_all_values:
#       ret_loss = loss
#     else:
#       ret_loss = loss.mean()
    #######

    return ret_loss
