def test_runner_cli_overrides_are_scoped_to_scheduling(monkeypatch, tmp_path):
    import yaml
    from memory_interference import runner

    cfg = {
        "data_file": str(tmp_path / "data.json"),
        "seed": 1,
        "num_keys": 2,
        "num_updates": [1],
        "episodes_per_level": 1,
        "queries_per_episode": 1,
        "models": [{"name": "m", "model_id": "id"}],
        "output_root": "outputs",
    }
    (tmp_path / "data.json").write_text('{"a": ["x", "y"], "b": ["u", "v"]}')
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    captured = {}

    def fake_load_model(spec):
        captured.update(spec)
        raise RuntimeError("stop after override inspection")

    monkeypatch.setattr(runner, "load_model", fake_load_model)
    try:
        runner.run(str(config_path), output_root=str(tmp_path / "isolated"), device="cuda:3")
    except RuntimeError as exc:
        assert str(exc) == "stop after override inspection"
    assert captured["device"] == "cuda:3"
    assert (tmp_path / "isolated").exists()
