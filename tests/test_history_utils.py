from src.utils.history import normalize_symbol


def test_normalize_symbol_for_plain_equity():
    assert normalize_symbol("tcs") == "TCS.NS"


def test_normalize_symbol_for_plain_forex_pair():
    assert normalize_symbol("usdinr") == "USDINR=X"


def test_normalize_symbol_preserves_explicit_symbols():
    assert normalize_symbol("USDINR=X") == "USDINR=X"
    assert normalize_symbol("TCS.NS") == "TCS.NS"
    assert normalize_symbol("^NSEI") == "^NSEI"
