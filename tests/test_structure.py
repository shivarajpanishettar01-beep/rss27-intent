from usintent.layout import check_all


def test_layout_contract() -> None:
    problems = check_all()
    assert not problems, "\n".join(problems)
