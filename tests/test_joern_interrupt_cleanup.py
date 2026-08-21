import pytest

from spectral_code.preprocessing import joern_runner


def test_progress_runner_terminates_child_tree_on_keyboard_interrupt(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = 130
            return self.returncode

    process = FakeProcess()
    terminated = []
    monkeypatch.setattr(joern_runner.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(joern_runner.time, "sleep", lambda seconds: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(joern_runner, "_terminate_process_tree", lambda current: terminated.append(current))

    with pytest.raises(KeyboardInterrupt):
        joern_runner._run_with_seconds_progress(
            ["fake-joern"],
            desc="Joern smoke",
            log_path=str(tmp_path / "parse.log"),
        )

    assert terminated == [process]
    assert process.returncode == 130
