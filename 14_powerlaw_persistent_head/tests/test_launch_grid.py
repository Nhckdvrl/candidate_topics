import os

import launch_grid


def test_visible_devices_respects_existing_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert launch_grid.visible_devices() == ["2", "3"]


def test_visible_devices_minus_one_disables_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert launch_grid.visible_devices() == []


def test_visible_devices_empty_disables_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert launch_grid.visible_devices() == []
