from time_r1.training.checkpointing import CheckpointManager


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(tmp_path, k=2)
    mgr.save(b"a", 0.1, 1)
    mgr.save(b"b", 0.2, 2)
    mgr.save(b"c", 0.05, 3)
    files = list(tmp_path.iterdir())
    assert len(files) == 2
    assert any("ckpt_2" in f.name for f in files)
