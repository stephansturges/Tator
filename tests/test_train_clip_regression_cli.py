from tools.train_clip_regression_from_YOLO import _print_matrix


def test_print_matrix_outputs_header_and_rows(capsys):
    _print_matrix([[2, 1], [0, 3]], ["car", "boat"])

    out = capsys.readouterr().out

    assert "true\\pred\tcar\tboat" in out
    assert "car\t2\t1" in out
    assert "boat\t0\t3" in out


def test_print_matrix_outputs_empty_message(capsys):
    _print_matrix([], [])

    out = capsys.readouterr().out

    assert "Confusion matrix is empty." in out
