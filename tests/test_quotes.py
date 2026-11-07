from lkybot.quotes import check_quotes


def test_exact_match_is_kept():
    answer = 'He said "We must be rugged."'
    cleaned, flagged = check_quotes(answer, ["We must be rugged."])
    assert flagged == []
    assert '"We must be rugged."' in cleaned


def test_curly_quotes_and_case_are_normalized():
    answer = "\u201ca nation must be ABLE to defend itself\u201d"
    cleaned, flagged = check_quotes(
        answer, ["A nation must be able to defend itself."]
    )
    assert flagged == []
    assert "defend itself" in cleaned


def test_fabricated_quote_is_flagged_and_removed():
    answer = 'He said "The moon is made of green cheese." That is my view.'
    cleaned, flagged = check_quotes(answer, ["The economy grew quickly."])
    assert flagged == ["The moon is made of green cheese."]
    assert "green cheese" not in cleaned
    assert "[quote removed: not found in sources]" in cleaned


def test_short_quote_is_ignored():
    answer = 'He said "three word quote" often.'
    cleaned, flagged = check_quotes(answer, ["The economy grew quickly."])
    assert flagged == []
    assert cleaned == answer
