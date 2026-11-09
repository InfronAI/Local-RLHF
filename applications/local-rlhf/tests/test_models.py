import torch

from local_rlhf.models import (load_policy, load_reward, load_tokenizer, reference_policy,
                           save_reward, snapshot_reference)


def test_reference_and_backbone_remain_frozen(tiny_model):
    model = load_policy(tiny_model, torch.device("cpu"), rank=4, checkpointing=False)
    snapshot_reference(model)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    ids = torch.tensor([[3, 16, 4, 14, 1]])
    with reference_policy(model):
        before = model(ids).logits.clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    with reference_policy(model):
        after = model(ids).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
    assert not torch.allclose(model(ids).logits, before)
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".reference." in n)


def test_reward_eos_padding_and_checkpoint(tiny_model, tmp_path):
    model = load_reward(tiny_model, torch.device("cpu"), rank=4, checkpointing=False).eval()
    tokenizer = load_tokenizer(tiny_model)
    ids = torch.tensor([[3, 16, 4, 14, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        score = model(ids, mask)
        torch.testing.assert_close(score, model(ids[:, :5], mask[:, :5]))
    save_reward(model, tokenizer, tmp_path)
    loaded = load_reward(tmp_path, torch.device("cpu"), checkpointing=False, require_head=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(score, loaded(ids, mask))
