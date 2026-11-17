from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NODE_TEST = ROOT / "tests" / "sit_to_stand_counter.test.js"


def test_sit_to_stand_counter_node_suite():
    node = shutil.which("node")
    assert node is not None, "Node.js is required for the browser counter unit tests"

    completed = subprocess.run(
        [node, "--test", str(NODE_TEST)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
