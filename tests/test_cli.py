import json

import pandas as pd

from forexgrand_core.cli import main


def test_preprocess_data_applies_function_and_writes_gzip_pickle(tmp_path, capsys):
    input_path = tmp_path / "prices.csv"
    input_path.write_text("close\n1.0\n2.5\n", encoding="utf-8")
    preprocess_path = tmp_path / "preprocess.py"
    preprocess_path.write_text(
        "def preprocess_fn(frame):\n"
        "    frame = frame.copy()\n"
        "    frame['delta'] = frame['close'].diff().fillna(0)\n"
        "    return frame\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "processed.pkl.gz"

    assert main([
        "preprocess_data",
        str(input_path),
        str(preprocess_path),
        "--output",
        str(output_path),
    ]) == 0

    result = pd.read_pickle(output_path, compression="gzip")
    assert result["delta"].tolist() == [0.0, 1.5]
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path)