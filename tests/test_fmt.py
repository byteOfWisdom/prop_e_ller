import propeller as p

def test_format_ev():
    assert p.ev(1.5, 0.3).format() == "1.5(3)"
    assert p.ev(15, 3).format() == "15(3)"
    # assert p.ev(15, 30).format() == ""
