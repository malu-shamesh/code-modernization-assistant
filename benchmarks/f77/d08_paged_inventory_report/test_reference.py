import importlib

import numpy as np
import target_module


def head1(page):
    return " PARTS INVENTORY" + " " * 20 + "PAGE " + str(page).rjust(3) + "\n"


HEAD2 = (
    " "
    + "ITEM".ljust(12)
    + "  "
    + "QTY".rjust(5)
    + "  "
    + "PRICE".rjust(9)
    + "  "
    + "VALUE".rjust(11)
    + "\n"
)
HEAD3 = " " + "-" * 43 + "\n"


def heading(page):
    return head1(page) + HEAD2 + HEAD3


def row(name, qty, price, ext):
    """One report row; price and ext are the hand formatted decimal texts."""
    return (
        " "
        + name.ljust(12)
        + "  "
        + str(qty).rjust(5)
        + "  "
        + price.rjust(9)
        + "  "
        + ext.rjust(11)
        + "\n"
    )


def test_block_data_default_page_length(capsys):
    # BLOCK DATA gives 55 lines per page and a page that starts full; a reload runs the
    # module-level initialisation again, so the test does not depend on the test order
    importlib.reload(target_module)
    for _ in range(52):
        target_module.prtlin("x")
    out = capsys.readouterr().out
    assert out.count("PARTS INVENTORY") == 1
    assert len(out.splitlines()) == 3 + 52
    target_module.prtlin("x")
    out = capsys.readouterr().out
    assert out == heading(2) + " x\n"


def test_main_prints_two_pages(capsys):
    target_module.main()
    out = capsys.readouterr().out
    page1 = (
        heading(1)
        + row("BOLT M8", 120, "0.35", "42.00")
        + row("WASHER 8", 400, "0.05", "20.00")
        + row("NUT M8", 150, "0.12", "18.00")
        + row("SPRING", 36, "2.50", "90.00")
        + row("GASKET", 12, "18.75", "225.00")
    )
    page2 = (
        heading(2)
        + row("PLATE 200", 5, "129.99", "649.95")
        + " "
        + "TOTAL".ljust(12)
        + "  "
        + "723".rjust(5)
        + " " * 13
        + "1044.95".rjust(11)
        + "\n"
        + " MEAN OVER  6 ITEMS"
        + " " * 3
        + "  0.1742E+03\n"
    )
    assert out == page1 + page2


def test_main_is_repeatable(capsys):
    target_module.main()
    first = capsys.readouterr().out
    target_module.main()
    assert capsys.readouterr().out == first


def test_prtlin_paginates(capsys):
    target_module.pginit(5)
    target_module.prtlin("hello")
    target_module.prtlin("world   ")
    target_module.prtlin("again")
    out = capsys.readouterr().out
    assert out == heading(1) + " hello\n" + " world\n" + heading(2) + " again\n"


def test_prtlin_blank_text_prints_one_blank(capsys):
    target_module.pginit(10)
    target_module.prtlin("     ")
    assert capsys.readouterr().out == heading(1) + "  \n"


def test_pginit_restarts_page_numbers(capsys):
    target_module.pginit(4)
    target_module.prtlin("a")
    target_module.prtlin("b")
    capsys.readouterr()
    target_module.pginit(4)
    target_module.prtlin("c")
    assert capsys.readouterr().out == heading(1) + " c\n"


def test_prtrow_formats(capsys):
    target_module.pginit(100)
    target_module.prtrow("BOLT M8", 120, 0.35)
    target_module.prtrow("LONG NAME THAT IS CUT", 7, 2.5)
    out = capsys.readouterr().out
    assert out == (
        heading(1) + row("BOLT M8", 120, "0.35", "42.00") + row("LONG NAME TH", 7, "2.50", "17.50")
    )


def test_prtrow_overflowing_fields_print_asterisks(capsys):
    target_module.pginit(100)
    target_module.prtrow("JIG", 1, 1000000.0)
    target_module.prtrow("BIG", 123456, 1.0)
    target_module.prtrow("EDGE", 1, 99999.99)
    out = capsys.readouterr().out
    assert out == (
        heading(1)
        + row("JIG", 1, "*********", "1000000.00")
        + row("BIG", "*****", "1.00", "123456.00")
        + row("EDGE", 1, "99999.99", "99999.99")
    )


def test_prtrow_fields_that_exactly_fill_the_width_are_not_overflow(capsys):
    target_module.pginit(100)
    target_module.prtrow("X", 12345, 1000.0)
    target_module.prtrow("Y", 1, 100000.0)
    out = capsys.readouterr().out
    assert out == (
        heading(1)
        + row("X", 12345, "1000.00", "12345000.00")
        + row("Y", 1, "100000.00", "100000.00")
    )


def test_prttot_small_mean_has_a_negative_exponent(capsys):
    target_module.pginit(100)
    target_module.prttot(1, 0.05, 1)
    out = capsys.readouterr().out
    assert out.endswith(" MEAN OVER  1 ITEMS" + " " * 3 + "  0.5000E-01\n")


def test_prttot_total_and_e_format_mean(capsys):
    target_module.pginit(100)
    target_module.prttot(3, 12345.678, 3)
    out = capsys.readouterr().out
    assert out == (
        heading(1)
        + " "
        + "TOTAL".ljust(12)
        + "  "
        + "3".rjust(5)
        + " " * 13
        + "12345.68".rjust(11)
        + "\n"
        + " MEAN OVER  3 ITEMS"
        + " " * 3
        + "  0.4115E+04\n"
    )


def test_prttot_negative_and_zero_mean(capsys):
    target_module.pginit(100)
    target_module.prttot(1, -0.5, 1)
    target_module.prttot(0, 0.0, 2)
    out = capsys.readouterr().out
    assert out == (
        heading(1)
        + " "
        + "TOTAL".ljust(12)
        + "  "
        + "1".rjust(5)
        + " " * 13
        + "-0.50".rjust(11)
        + "\n"
        + " MEAN OVER  1 ITEMS"
        + " " * 3
        + " -0.5000E+00\n"
        + " "
        + "TOTAL".ljust(12)
        + "  "
        + "0".rjust(5)
        + " " * 13
        + "0.00".rjust(11)
        + "\n"
        + " MEAN OVER  2 ITEMS"
        + " " * 3
        + "  0.0000E+00\n"
    )


def test_prttot_without_items_has_no_mean_line(capsys):
    target_module.pginit(100)
    target_module.prttot(10, 99.5, 0)
    out = capsys.readouterr().out
    assert out == (
        heading(1)
        + " "
        + "TOTAL".ljust(12)
        + "  "
        + "10".rjust(5)
        + " " * 13
        + "99.50".rjust(11)
        + "\n"
    )


def field(text):
    return text.rjust(9)


def test_prtmat_format_reversion(capsys):
    a = np.zeros((3, 8))
    for i in range(1, 4):
        for j in range(1, 9):
            a[i - 1, j - 1] = 10 * i + j
    target_module.prtmat(a, 3, 2, 8)
    out = capsys.readouterr().out
    expected = (
        " ROW  1:"
        + "".join(field(f"{v}.00") for v in (11, 12, 13, 14, 15, 16))
        + "\n"
        + field("17.00")
        + field("18.00")
        + "\n"
        + " ROW  2:"
        + "".join(field(f"{v}.00") for v in (21, 22, 23, 24, 25, 26))
        + "\n"
        + field("27.00")
        + field("28.00")
        + "\n"
    )
    assert out == expected


def test_prtmat_reverts_to_the_group_again_for_a_third_record(capsys):
    a = np.arange(1.0, 14.0).reshape(1, 13)
    target_module.prtmat(a, 1, 1, 13)
    expected = (
        " ROW  1:"
        + "".join(field(f"{v}.00") for v in range(1, 7))
        + "\n"
        + "".join(field(f"{v}.00") for v in range(7, 13))
        + "\n"
        + field("13.00")
        + "\n"
    )
    assert capsys.readouterr().out == expected


def test_prtmat_uses_leading_dimension_and_ignores_the_rest(capsys):
    a = np.arange(1.0, 13.0).reshape(4, 3, order="F")
    # column major: A(1,1)=1, A(2,1)=2, A(3,1)=3, A(4,1)=4, A(1,2)=5, ...
    target_module.prtmat(a, 4, 2, 3)
    out = capsys.readouterr().out
    assert out == (
        " ROW  1:"
        + field("1.00")
        + field("5.00")
        + field("9.00")
        + "\n"
        + " ROW  2:"
        + field("2.00")
        + field("6.00")
        + field("10.00")
        + "\n"
    )


def test_prtmat_exactly_six_columns_has_no_continuation_line(capsys):
    a = np.full((1, 6), 0.5)
    target_module.prtmat(a, 1, 1, 6)
    assert capsys.readouterr().out == " ROW  1:" + field("0.50") * 6 + "\n"


def test_prtmat_zero_columns_and_zero_rows(capsys):
    a = np.ones((2, 2))
    target_module.prtmat(a, 2, 2, 0)
    assert capsys.readouterr().out == " ROW  1:\n ROW  2:\n"
    target_module.prtmat(a, 2, 0, 2)
    assert capsys.readouterr().out == ""
