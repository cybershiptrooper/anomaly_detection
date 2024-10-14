import torch

from collections import defaultdict


# Function to add hooks and cache activations
def add_hooks_and_cache(model):
    activation_cache = defaultdict(list)

    def hook_fn(module, input, output, name):
        activation_cache[name].append(output.detach())

    hooks = []
    for name, module in model.model.named_modules():
        if "mlp" in name and "down_proj" in name:
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))
            activation_cache[name] = []

    return activation_cache, hooks


def make_diff_cache(
    model,
    p1: str,
    p2: str,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = model.tokenizer
    input_ids_p1 = tokenizer.encode(p1)
    input_ids_p2 = tokenizer.encode(p2)

    activation_cache_p1, hooks_p1 = add_hooks_and_cache(model)

    # Forward pass
    # response1 = model.generate(p1)
    model.model.forward(torch.tensor(input_ids_p1).unsqueeze(0).to(device))

    for hook in hooks_p1:
        hook.remove()

    activation_cache_p2, hooks_p2 = add_hooks_and_cache(model)

    model.model.forward(torch.tensor(input_ids_p2).unsqueeze(0).to(device))

    # response2 = model.generate(p2)
    for hook in hooks_p2:
        hook.remove()

    diff_cache = {}
    for key in activation_cache_p1.keys():
        diff_vector = activation_cache_p1[key][0][:, -1] - activation_cache_p2[key][0][:, -1]
        # normalize
        diff_vector = diff_vector / (torch.norm(diff_vector, dim=1) + 1e-6)
        diff_cache[key] = diff_vector.squeeze(0)

    return diff_cache


def make_diff_cache_for_list_of_prompts(
    model,
    p1: list[str],
    p2: list[str],
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = model.tokenizer
    activation_cache_p1 = defaultdict(list)
    activation_cache_p2 = defaultdict(list)

    for prompt1, prompt2 in zip(p1, p2):
        input_ids_p1 = tokenizer.encode(prompt1)
        input_ids_p2 = tokenizer.encode(prompt2)

        cache_p1, hooks_p1 = add_hooks_and_cache(model)
        model.model.forward(torch.tensor(input_ids_p1).unsqueeze(0).to(device))
        for hook in hooks_p1:
            hook.remove()

        cache_p2, hooks_p2 = add_hooks_and_cache(model)
        model.model.forward(torch.tensor(input_ids_p2).unsqueeze(0).to(device))
        for hook in hooks_p2:
            hook.remove()

        for key in cache_p1.keys():
            activation_cache_p1[key].append(cache_p1[key][0][:, -1])
            activation_cache_p2[key].append(cache_p2[key][0][:, -1])

    diff_cache = {}
    for key in activation_cache_p1.keys():
        diff_vectors = [
            p1 - p2 for p1, p2 in zip(activation_cache_p1[key], activation_cache_p2[key])
        ]
        mean_diff_vector = torch.mean(torch.stack(diff_vectors), dim=0)
        # normalize
        mean_diff_vector = mean_diff_vector / (torch.norm(mean_diff_vector, dim=1) + 1e-6)
        diff_cache[key] = mean_diff_vector.squeeze(0)

    return diff_cache
