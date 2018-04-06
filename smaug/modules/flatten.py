from torch.nn import Module


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.view(inp.size(0), -1)
