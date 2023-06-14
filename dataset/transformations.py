class PermuteTransform:
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        return sample.permute(self.permutation)
