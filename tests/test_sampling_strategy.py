from time_r1.training.sampling_strategy import ClusterRandomSampler, LocalRandomSampler


def test_local_random_sampler():
    items = list(range(10))
    sampler = LocalRandomSampler(radius=1)
    sample = sampler.sample(items, 3)
    assert len(sample) == 3
    for s in sample:
        assert s in items


def test_cluster_random_sampler():
    items = list(range(20))
    sampler = ClusterRandomSampler(clusters=4)
    sample = sampler.sample(items, 4)
    assert len(sample) == 4
    for s in sample:
        assert s in items
