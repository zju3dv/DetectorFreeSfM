import torch
from loguru import logger

# backward compatibility
STATE_DICT_MAPPER = {
    'coarse_matching.skh_bin_score': 'coarse_matching.optimal_transport.bin_score',
    'coarse_matching.skh_bin_prototype': 'coarse_matching.optimal_transport.bin_prototype',
    'fine_preprocess.down_proj.0.weight': 'fine_preprocess.down_proj.weight',
    'fine_preprocess.down_proj.0.bias': 'fine_preprocess.down_proj.bias',
    "pos_encoding.pe": None
}

def update_state_dict(mapper,
                      ckpt_path=None,
                      state_dict=None,
                      save=False):
    updated = False
    if state_dict is None:
        state_dict = torch.load(ckpt_path)
    for old_key, new_key in mapper.items():
        if old_key in state_dict:
            updated = True
            if new_key is not None:
                state_dict[new_key] = state_dict.pop(old_key)
            else:
                logger.warning(f"Key: {old_key} is deleated when loading loftr model")
                state_dict.pop(old_key)

    if updated and save:
        torch.save(state_dict, ckpt_path)
    return state_dict, updated

@torch.no_grad()
def torch_speed_test(model, args, kwargs={}, n_rounds=100, model_name='Model'):
    # init records
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # warmup
    model(*args, **kwargs)
    torch.cuda.synchronize()
    # timing
    start_event.record()
    for _ in range(n_rounds):
        _ = model(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    print(f"[{model_name}]: {start_event.elapsed_time(end_event) / n_rounds} ms.")

