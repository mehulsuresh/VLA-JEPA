from starVLA.training.trainer_utils.trainer_tools import TrainerUtils


class _EpochAwareLoader:
    def __init__(self):
        self.epochs = []
        self.iterations = 0

    def set_epoch(self, epoch):
        self.epochs.append(epoch)

    def __iter__(self):
        self.iterations += 1
        return iter(())


class _Sampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class _SamplerOnlyLoader:
    def __init__(self):
        self.sampler = _Sampler()

    def __iter__(self):
        return iter(())


def test_reset_dataloader_prefers_wrapper_epoch_management():
    loader = _EpochAwareLoader()

    iterator, epoch = TrainerUtils._reset_dataloader(loader, 2)

    assert list(iterator) == []
    assert epoch == 3
    assert loader.epochs == [3]
    assert loader.iterations == 1


def test_reset_dataloader_falls_back_to_sampler_set_epoch():
    loader = _SamplerOnlyLoader()

    iterator, epoch = TrainerUtils._reset_dataloader(loader, 4)

    assert list(iterator) == []
    assert epoch == 5
    assert loader.sampler.epochs == [5]
