import torch


def requires_grad_is_False():
    a = torch.rand((2, 3), dtype=torch.float, requires_grad=True)
    x = 2 * a
    with torch.no_grad():
        w = a * a
    y = (x * w).sum()
    y.backward()
    assert (torch.all(torch.abs(a.grad - 2 * w) < 1e-6))


def normal_case():
    torch.manual_seed(0)
    a = torch.rand((2, 3), dtype=torch.float, requires_grad=True)
    x = 2 * a
    w = a * a
    y = (x * w).sum()
    y.backward()
    assert (torch.all(torch.abs(a.grad - 6 * a * a) < 1e-6))


if __name__ == "__main__":
    normal_case()
    requires_grad_is_False()
