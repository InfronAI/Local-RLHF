import torch

from local_rlhf.objectives import (clipped_policy_loss, clipped_value_loss, dpo_loss,
                               generalized_advantages, token_log_probs)


def test_gae_terminal_returns():
    advantage, returns = generalized_advantages(torch.tensor([1., 2.]), torch.tensor([.5, .25]), 1., 1.)
    torch.testing.assert_close(returns, torch.tensor([3., 2.]))
    torch.testing.assert_close(advantage, torch.tensor([2.5, 1.75]))


def test_ppo_policy_and_value_clipping():
    loss = clipped_policy_loss(torch.log(torch.tensor([2., .5])), torch.zeros(2), torch.tensor([1., -1.]), .2)
    torch.testing.assert_close(loss, torch.tensor(-.2))
    value = clipped_value_loss(torch.tensor([3.]), torch.tensor([0.]), torch.tensor([1.]), .2)
    torch.testing.assert_close(value, torch.tensor(2.))


def test_dpo_prefers_chosen_and_has_finite_gradient():
    chosen = torch.tensor([-2.], requires_grad=True)
    rejected = torch.tensor([-4.], requires_grad=True)
    loss = dpo_loss(chosen, rejected, torch.tensor([-3.]), torch.tensor([-3.]), .1)
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert loss.item() < torch.log(torch.tensor(2.)).item()


def test_log_probs_have_correct_causal_shift():
    logits = torch.tensor([[[0., 2., 0.], [2., 0., 0.], [0., 0., 0.]]])
    ids = torch.tensor([[2, 1, 0]])
    actual = token_log_probs(logits, ids)
    expected = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    torch.testing.assert_close(actual, expected)
