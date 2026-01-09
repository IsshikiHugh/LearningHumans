import torch

def look_tensor(
    x      : torch.Tensor,
    silent : bool = False,
):
    """
    Summarize the information of a tensor, including its shape, value range (min, max, mean, std), and dtype.
    Then return a string containing the information.

    ### Args
    - `x`: torch.Tensor
    - `silent`: bool, default `False`
        - If not silent, the function will print the message itself. The information string will always be returned.

    ### Returns
    - str
    """
    info_list = []
    info_list.append(f'shape = {tuple(x.shape)}')
    info_list.append(f'dtype = {str(x.dtype)}')
    info_list.append(f'device = {str(x.device)}')
    # Convert to float to calculate the statistics.
    x = x.float()
    info_list.append(f'min/max/mean/std = [ {x.min():06f} -> {x.max():06f} ] ~ ( {x.mean():06f}, {x.std():06f} )')
    # Generate the final information and print it if necessary.
    ret = '\t'.join(info_list)
    if not silent:
        print(ret)
    return ret