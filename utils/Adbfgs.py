import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class Adbfgs(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99, 0.92,0.0), rho = 0.04,
         weight_decay=1e-1, *, maximize: bool = False,
         capturable: bool = False,
         if_acc: bool = True
         ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho, 
                        weight_decay=weight_decay, 
                        maximize=maximize, capturable=capturable)
        super(Adbfgs, self).__init__(params, defaults)
        

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))
    
    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2, beta3, beta4 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)


    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = [] # g
            exp_avg_diffs = [] # y
            exp_avg_ss = [] #s
            exp_avg_hs = [] # h
            neg_pre_grads = [] # for y
            state_steps = []
            # hessian = []
            beta1, beta2, beta3, beta4= group['betas']
            
            

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_diff'] = torch.zeros_like(p,memory_format=torch.preserve_format) 
                    state['exp_avg_s'] = torch.zeros_like(p,memory_format=torch.preserve_format) 
                    state['exp_avg_h'] = torch.zeros_like(p,memory_format=torch.preserve_format) 
                    state['neg_pre_grad'] = torch.zeros_like(p,memory_format=torch.preserve_format) 
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)                

                # if 'neg_pre_grad' not in state.keys():
                    # state['neg_pre_grad'] = torch.zeros_like(p,memory_format=torch.preserve_format)  
                state_steps.append(state['step'])

                exp_avgs.append(state['exp_avg'])
                # hessian.append(state['hessian'])
                exp_avg_diffs.append(state['exp_avg_diff'])
                exp_avg_ss.append(state['exp_avg_s'])
                exp_avg_hs.append(state['exp_avg_h'])
                neg_pre_grads.append(state['neg_pre_grad'])
                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            adbfgs(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_diffs,
                  exp_avg_ss,
                  exp_avg_hs,
                  neg_pre_grads,
                  state_steps,
                  bs=bs,
                  beta1=beta1,
                  beta2=beta2,
                  beta3=beta3,
                  beta4=beta4,
                  rho=group['rho'],
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  maximize=group['maximize'],
                  capturable=group['capturable'])

        return loss

def adbfgs(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_diffs:  List[Tensor],
          exp_avg_ss:  List[Tensor],
          exp_avg_hs:  List[Tensor],
          neg_pre_grads:  List[Tensor],
          state_steps: List[Tensor],
          capturable: bool = False,
          *,
          bs: int,
          beta1: float,
          beta2: float,
          beta3: float,
          beta4: float,
          rho: float,
          lr: float,
          weight_decay: float,
          maximize: bool):

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    
    func = _single_tensor_adbfgs

    func(params,
         grads,
         exp_avgs,
         exp_avg_diffs,
         exp_avg_ss,
         exp_avg_hs,
         neg_pre_grads,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         beta3=beta3,
         beta4=beta4,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)

def _single_tensor_adbfgs(params: List[Tensor],
                         grads: List[Tensor],
                         exp_avgs: List[Tensor],
                         exp_avg_diffs: List[Tensor],
                         exp_avg_ss: List[Tensor],
                         exp_avg_hs: List[Tensor],
                         neg_pre_grads: List[Tensor],
                         state_steps: List[Tensor],
                         *,
                         bs: int,
                         beta1: float,
                         beta2: float,
                         beta3: float,
                         beta4: float,
                         rho: float,
                         lr: float,
                         weight_decay: float,
                         maximize: bool,
                         capturable: bool):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_diff = exp_avg_diffs[i]
        exp_avg_s = exp_avg_ss[i]
        exp_avg_h = exp_avg_hs[i]
        neg_grad_or_diff = neg_pre_grads[i]
        # hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda 
            
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_s = torch.view_as_real(exp_avg_s)
            exp_avg_h = torch.view_as_real(exp_avg_h)
            exp_avg_diff = torch.view_as_real(exp_avg_diff)
            neg_grad_or_diff = torch.view_as_real(neg_grad_or_diff)
            # hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        
        #
        # grad.mul(torch.sign(param))
        
        # Decay the first and second moment running average coefficient
        neg_grad_or_diff.add_(grad)
        
        # exp_avg_diff.mul_(beta3).add_(grad,alpha=1 - beta3).add_(exp_avg,alpha= beta3-1)
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_diff.mul_(beta3).add_(neg_grad_or_diff, alpha=1 - beta3)
        neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)

        # neg_grad_or_diff.mul_(beta3).add_(grad)
        # exp_avg_diff.mul_(beta2).add_(exp_avg,alpha=-1)
        if capturable:
            step = step_t
            step_size = lr 
            step_size_neg = step_size.neg()
            tmp = exp_avg + exp_avg_diff *0.5#(1 - beta3)
            ratio = (tmp.abs() / (rho * 1 * exp_avg_h + 1e-15)).clamp(None,1)
            # tmp = torch.exp(step_size_neg * tmp).clamp(None,1.1)
            # param.mul_(tmp)#.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            # param.mul_(torch.exp(step_size_neg * tmp.sign() * ratio))
            
            # sum_param = torch.sum(param.abs())
            # print(sum_param)
            # if sum_param > 3e5:
            #    param.div_(sum_param).mul_(3e5)
            exp_avg_s.mul_(beta4).addcmul_(tmp.sign(), ratio, value=step_size_neg*(1-beta4))
            # param.addcmul_(tmp.sign(), ratio, value=step_size_neg)

            tmp_val = torch.abs(torch.sum(exp_avg_diff*exp_avg_s))+1e-15
            # exp_avg_h.mul_(beta2).addcmul_(exp_avg_diff,exp_avg_diff,value = (1-beta2))
            exp_avg_h.mul_(beta2).add(torch.abs(exp_avg_diff - exp_avg_h*exp_avg_s),alpha = 1-beta2)
            param.add_(exp_avg_s)

            
            # ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            # param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr 
            
            tmp = exp_avg + exp_avg_diff *0.5# (1 - beta3)
            ratio = (tmp.abs() / (rho * 1 * exp_avg_h + 1e-15)).clamp(None,1)
            exp_avg_s.mul_(beta4).addcmul_(tmp.sign(), ratio, value=step_size_neg*(1-beta4))
            # param.addcmul_(tmp.sign(), ratio, value=step_size_neg)
            tmp_val = torch.abs(torch.sum(exp_avg_diff*exp_avg_s))
            # exp_avg_h.mul_(beta2).addcmul_(exp_avg_diff,exp_avg_diff,value = (1-beta2))
            exp_avg_h.mul_(beta2).add(torch.abs(exp_avg_diff - exp_avg_h*exp_avg_s),alpha = 1-beta2)
            param.add_(exp_avg_s)
            # tmp = torch.exp(step_size_neg * tmp ).clamp(None,1.1)
            # param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            
            
                        
            # sum_param = torch.sum(param.abs())
            
            # if sum_param > 3e5:
            #    param.div_(sum_param).mul_(3e5)
            # param.addcmul_(tmp.sign(), ratio, value=step_size_neg)
            # ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None,1)
            # param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
            
        