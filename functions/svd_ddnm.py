import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import pdb

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, adapter, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None, args=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        if args.time_travel:
            for i, j in tqdm(time_pairs):
                i, j = i*skip, j*skip
                if j<0: j=-1 

                if j < i: # normal sampling 
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    xt = xs[-1].to('cuda')

                    # if front: xt = xt - adapter(xt), then put in model
                    if cls_fn == None:
                        et = model(xt, t)  # original
                    else:
                        classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                        et = model(xt, t, classes)  # original
                        et = et[:, :3]
                        et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                    if et.size(1) == 6:
                        et = et[:, :3]

                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    # if behind: (this is svd & sigma=0, do not forget modify my_sample in diffusion.py)
                    if args.A_type == "math" or "unet" in args.A_type:
                        x0_t_hat = x0_t - A_funcs.A_pinv(
                            A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                        ).reshape(*x0_t.size())
                    elif args.A_type == "adapter":
                        x0_t_hat = x0_t - adapter(A_funcs.A_pinv(
                            A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                        ).reshape(*x0_t.size()), t)

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                    xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                else: # time-travel back
                    next_t = (torch.ones(n) * j).to(x.device)
                    at_next = compute_alpha(b, next_t.long())
                    x0_t = x0_preds[-1].to('cuda')
                    
                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                    xs.append(xt_next.to('cpu'))
        else:
            for i, j in tqdm(time_pairs):
                i, j = i*skip, j*skip
                if j<0: j=-1 

                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                if args.A_type == "math":
                    x0_t_hat = x0_t - A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(*x0_t.size())
                elif args.A_type == "adapter":
                    x0_t_hat = x0_t - adapter(A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(*x0_t.size()), t)
                
                etaB=0.4
                etaA=1
                xt_next = (etaB * at_next.sqrt() * x0_t_hat + (1 - etaB) * at_next.sqrt() * x0_t + 
                etaA * (1 - at_next).sqrt() * torch.randn_like(x0_t_hat) + (1 - etaA) * et * (1 - at_next).sqrt())

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def ddnm_plus_diffusion(x, model, adapter, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, args=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))        
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
                if args.A_type == "math":
                    x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())
                elif args.A_type == "adapter":
                    x0_t_hat = x0_t - adapter(A_funcs.Lambda(A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size()), t)


                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
