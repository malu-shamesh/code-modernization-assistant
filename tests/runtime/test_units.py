"""File units: open/close semantics, sequential positioning, implicit units and thread safety."""

from __future__ import annotations

import io
import sys
import threading
from pathlib import Path

import pytest

from modernizer.runtime import f77rt as rt


@pytest.fixture(autouse=True)
def clean_units(tmp_path, monkeypatch):
    """Run in an empty directory (implicit fort.N files land here) with no open units."""
    monkeypatch.chdir(tmp_path)
    rt.close_all_units()
    yield
    rt.close_all_units()


def lines(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8").split("\n")


class TestWritingFiles:
    def test_formatted_and_list_directed_records(self, tmp_path):
        path = tmp_path / "out.txt"
        rt.open_unit(10, path, status="replace")
        rt.write_formatted(10, "(I3)", 1, 2)
        rt.write_list(10, "x")
        rt.close_unit(10)
        assert path.read_bytes() == b"  1\n  2\n x\n"

    def test_data_is_visible_before_close(self, tmp_path):
        path = tmp_path / "out.txt"
        rt.open_unit(10, path, status="replace")
        rt.write_formatted(10, "(A)", "abc")
        assert path.read_bytes() == b"abc\n"

    def test_file_name_may_be_blank_padded_string(self, tmp_path):
        path = tmp_path / "padded.txt"
        rt.open_unit(11, str(path) + "   ", status="replace")
        rt.write_formatted(11, "(A)", "ok")
        rt.close_unit(11)
        assert path.read_bytes() == b"ok\n"

    def test_unicode_text_round_trips(self, tmp_path):
        path = tmp_path / "u.txt"
        rt.open_unit(10, path, status="replace")
        rt.write_formatted(10, "(A)", "café")
        rt.rewind_unit(10)
        assert rt.read_formatted(10, "(A)") == ["café"]

    def test_unit_6_can_be_reconnected_to_a_file(self, tmp_path, capsys):
        path = tmp_path / "six.txt"
        rt.open_unit(6, path, status="replace")
        rt.write_formatted(6, "(A)", "to file")
        rt.close_unit(6)
        rt.write_formatted(6, "(A)", "to screen")
        assert path.read_bytes() == b"to file\n"
        assert capsys.readouterr().out == "to screen\n"

    def test_format_errors_leave_the_file_untouched(self, tmp_path):
        path = tmp_path / "out.txt"
        rt.open_unit(10, path, status="replace")
        with pytest.raises(ValueError):
            rt.write_formatted(10, "(I3,F)", 1, 2.0)
        assert path.read_bytes() == b""


class TestReadingFiles:
    def make(self, tmp_path, text=b"  1\n  2\n  3\n"):
        path = tmp_path / "in.txt"
        path.write_bytes(text)
        rt.open_unit(12, path, status="old")
        return path

    def test_sequential_reads_then_end_of_file(self, tmp_path):
        self.make(tmp_path)
        assert [rt.read_formatted(12, "(I3)")[0] for _ in range(3)] == [1, 2, 3]
        with pytest.raises(EOFError):
            rt.read_formatted(12, "(I3)")
        with pytest.raises(EOFError):
            rt.read_formatted(12, "(I3)")

    def test_rewind_restarts(self, tmp_path):
        self.make(tmp_path)
        rt.read_formatted(12, "(I3)")
        rt.read_formatted(12, "(I3)")
        rt.rewind_unit(12)
        assert rt.read_formatted(12, "(I3)") == [1]

    def test_backspace_rereads_the_previous_record(self, tmp_path):
        self.make(tmp_path)
        rt.read_formatted(12, "(I3)")
        assert rt.read_formatted(12, "(I3)") == [2]
        rt.backspace_unit(12)
        assert rt.read_formatted(12, "(I3)") == [2]
        rt.backspace_unit(12)
        rt.backspace_unit(12)
        assert rt.read_formatted(12, "(I3)") == [1]

    def test_backspace_at_the_first_record_does_nothing(self, tmp_path):
        self.make(tmp_path)
        rt.backspace_unit(12)
        assert rt.read_formatted(12, "(I3)") == [1]

    def test_backspace_after_end_of_file_returns_before_the_endfile_record(self, tmp_path):
        self.make(tmp_path, b"  1\n  2\n")
        rt.read_formatted(12, "(I3)")
        rt.read_formatted(12, "(I3)")
        with pytest.raises(EOFError):
            rt.read_formatted(12, "(I3)")
        rt.backspace_unit(12)  # now positioned after the last data record again
        with pytest.raises(EOFError):
            rt.read_formatted(12, "(I3)")
        rt.backspace_unit(12)
        rt.backspace_unit(12)
        assert rt.read_formatted(12, "(I3)") == [2]

    def test_windows_line_endings(self, tmp_path):
        self.make(tmp_path, b"1\r\n2\r\n")
        assert rt.read_formatted(12, "(A)") == ["1"]
        assert rt.read_formatted(12, "(A)") == ["2"]

    def test_last_line_without_newline(self, tmp_path):
        self.make(tmp_path, b"a\nb")
        assert rt.read_formatted(12, "(A)") == ["a"]
        assert rt.read_formatted(12, "(A)") == ["b"]
        with pytest.raises(EOFError):
            rt.read_formatted(12, "(A)")

    def test_list_directed_read_spans_records(self, tmp_path):
        self.make(tmp_path, b"1 2\n3\n4\n")
        assert rt.read_list(12, 3) == [1, 2, 3]
        assert rt.read_list(12, 1, int) == [4]

    def test_read_with_reversion_from_a_file(self, tmp_path):
        self.make(tmp_path, b"1\n2\n3\n")
        assert rt.read_formatted(12, "(I1)", n=3) == [1, 2, 3]

    def test_read_only_files_can_be_read_and_never_damaged(self, tmp_path):
        path = tmp_path / "ro.txt"
        path.write_bytes(b"x\n")
        path.chmod(0o444)
        try:
            rt.open_unit(13, path, status="old")
            assert rt.read_formatted(13, "(A)") == ["x"]
            if rt._UNITS[13].read_only:
                with pytest.raises(OSError):
                    rt.write_formatted(13, "(A)", "y")
        finally:
            rt.close_all_units()
            path.chmod(0o666)
        assert path.read_bytes() == b"x\n"


class TestSequentialSemantics:
    def test_writing_after_rewind_truncates(self, tmp_path):
        path = tmp_path / "t.txt"
        rt.open_unit(14, path, status="replace")
        rt.write_formatted(14, "(I1)", 1, 2, 3)
        rt.rewind_unit(14)
        rt.write_formatted(14, "(I1)", 9)
        rt.close_unit(14)
        assert path.read_bytes() == b"9\n"

    def test_writing_in_the_middle_destroys_the_rest(self, tmp_path):
        path = tmp_path / "t.txt"
        rt.open_unit(14, path, status="replace")
        rt.write_formatted(14, "(I1)", 1, 2, 3)
        rt.rewind_unit(14)
        rt.read_formatted(14, "(I1)")
        rt.write_formatted(14, "(I1)", 8)
        rt.close_unit(14)
        assert path.read_bytes() == b"1\n8\n"

    def test_backspace_then_overwrite_the_last_record(self, tmp_path):
        path = tmp_path / "t.txt"
        rt.open_unit(14, path, status="replace")
        rt.write_formatted(14, "(I1)", 1, 2, 3)
        rt.backspace_unit(14)
        rt.write_formatted(14, "(I1)", 7)
        rt.close_unit(14)
        assert path.read_bytes() == b"1\n2\n7\n"

    def test_endfile_truncates_at_the_current_position(self, tmp_path):
        path = tmp_path / "t.txt"
        rt.open_unit(14, path, status="replace")
        rt.write_formatted(14, "(I1)", 1, 2, 3)
        rt.rewind_unit(14)
        rt.read_formatted(14, "(I1)")
        rt.endfile_unit(14)
        assert path.read_bytes() == b"1\n"
        with pytest.raises(EOFError):
            rt.read_formatted(14, "(I1)")

    def test_write_rewind_read_cycle(self, tmp_path):
        rt.open_unit(15, tmp_path / "cycle.txt", status="replace")
        rt.write_formatted(15, "(F6.2,I3)", 3.14159, 7)
        rt.rewind_unit(15)
        assert rt.read_formatted(15, "(F6.2,I3)") == [pytest.approx(3.14), 7]

    def test_position_append(self, tmp_path):
        path = tmp_path / "log.txt"
        path.write_bytes(b"a\nb\n")
        rt.open_unit(16, path, status="old", position="append")
        rt.write_formatted(16, "(A)", "c")
        rt.backspace_unit(16)
        rt.write_formatted(16, "(A)", "d")
        rt.close_unit(16)
        assert path.read_bytes() == b"a\nb\nd\n"

    def test_rewind_and_backspace_on_standard_units_do_nothing(self):
        for unit in (0, 5, 6):
            rt.rewind_unit(unit)
            rt.backspace_unit(unit)
            rt.endfile_unit(unit)


class TestOpenStatus:
    def test_old_requires_an_existing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rt.open_unit(20, tmp_path / "missing.txt", status="old")

    def test_new_requires_a_missing_file(self, tmp_path):
        path = tmp_path / "n.txt"
        rt.open_unit(20, path, status="new")
        rt.close_unit(20)
        assert path.exists()
        with pytest.raises(FileExistsError):
            rt.open_unit(21, path, status="new")

    def test_replace_empties_an_existing_file(self, tmp_path):
        path = tmp_path / "r.txt"
        path.write_bytes(b"old\n")
        rt.open_unit(20, path, status="replace")
        rt.close_unit(20)
        assert path.read_bytes() == b""

    def test_unknown_creates_or_keeps(self, tmp_path):
        path = tmp_path / "u.txt"
        rt.open_unit(20, path)
        rt.write_formatted(20, "(A)", "first")
        rt.close_unit(20)
        rt.open_unit(20, path, status="unknown")
        assert rt.read_formatted(20, "(A)") == ["first"]

    def test_status_is_case_insensitive(self, tmp_path):
        rt.open_unit(
            20, tmp_path / "c.txt", status="REPLACE", form="FORMATTED", access="Sequential"
        )

    def test_scratch_file_is_deleted_on_close(self, tmp_path):
        rt.open_unit(22, status="scratch")
        scratch_path = Path(rt._UNITS[22].path)
        rt.write_formatted(22, "(I2)", 5)
        rt.rewind_unit(22)
        assert rt.read_formatted(22, "(I2)") == [5]
        assert scratch_path.exists()
        rt.close_unit(22)
        assert not scratch_path.exists()

    def test_scratch_file_must_not_be_named(self, tmp_path):
        with pytest.raises(ValueError):
            rt.open_unit(22, tmp_path / "s.txt", status="scratch")

    def test_close_with_delete_removes_the_file(self, tmp_path):
        path = tmp_path / "d.txt"
        rt.open_unit(23, path, status="replace")
        rt.close_unit(23, status="delete")
        assert not path.exists()

    def test_close_status_keep_and_validation(self, tmp_path):
        path = tmp_path / "k.txt"
        rt.open_unit(23, path, status="replace")
        with pytest.raises(ValueError):
            rt.close_unit(23, status="bogus")
        rt.close_unit(23, status="keep")
        assert path.exists()
        rt.open_unit(24, status="scratch")
        with pytest.raises(ValueError):
            rt.close_unit(24, status="keep")
        rt.close_unit(24)

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"status": "bogus"}, ValueError),
            ({"form": "binary"}, ValueError),
            ({"form": "unformatted"}, NotImplementedError),
            ({"access": "stream"}, ValueError),
            ({"access": "direct"}, NotImplementedError),
            ({"position": "middle"}, ValueError),
        ],
    )
    def test_invalid_or_unsupported_specifiers(self, tmp_path, kwargs, exc):
        with pytest.raises(exc):
            rt.open_unit(25, tmp_path / "x.txt", **kwargs)

    def test_unit_numbers_are_validated(self, tmp_path):
        with pytest.raises(ValueError):
            rt.open_unit(-1, tmp_path / "x.txt")
        with pytest.raises(TypeError):
            rt.open_unit("ten", tmp_path / "x.txt")
        with pytest.raises(TypeError):
            rt.write_formatted(2.5, "(I1)", 1)

    def test_closing_an_unconnected_unit_is_allowed(self):
        rt.close_unit(99)

    def test_same_file_cannot_be_connected_twice(self, tmp_path):
        path = tmp_path / "shared.txt"
        rt.open_unit(30, path, status="replace")
        with pytest.raises(ValueError):
            rt.open_unit(31, path)

    @pytest.mark.skipif(
        sys.platform != "win32", reason="file names are case-insensitive on Windows"
    )
    def test_same_file_with_different_letter_case_is_detected(self, tmp_path):
        path = tmp_path / "Case.txt"
        rt.open_unit(30, path, status="replace")
        with pytest.raises(ValueError):
            rt.open_unit(31, str(path).swapcase())

    def test_reopening_the_same_unit_and_file_keeps_the_connection(self, tmp_path):
        path = tmp_path / "keep.txt"
        rt.open_unit(30, path, status="replace")
        rt.write_formatted(30, "(A)", "one")
        rt.open_unit(30, path)
        rt.write_formatted(30, "(A)", "two")
        rt.open_unit(30)
        rt.close_unit(30)
        assert path.read_bytes() == b"one\ntwo\n"

    def test_reopening_a_unit_with_another_file_closes_the_first(self, tmp_path):
        first, second = tmp_path / "first.txt", tmp_path / "second.txt"
        rt.open_unit(30, first, status="replace")
        rt.write_formatted(30, "(A)", "one")
        rt.open_unit(30, second, status="replace")
        rt.write_formatted(30, "(A)", "two")
        rt.close_unit(30)
        assert first.read_bytes() == b"one\n"
        assert second.read_bytes() == b"two\n"

    def test_close_all_units(self, tmp_path):
        for unit in (40, 41):
            rt.open_unit(unit, tmp_path / f"f{unit}.txt", status="replace")
        rt.close_all_units()
        assert rt._UNITS == {}


class TestImplicitUnits:
    def test_write_to_an_unconnected_unit_creates_fort_n(self, tmp_path):
        rt.write_formatted(50, "(I3)", 7)
        rt.close_unit(50)
        assert (tmp_path / "fort.50").read_bytes() == b"  7\n"

    def test_read_from_an_unconnected_unit_needs_an_existing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rt.read_formatted(51, "(I3)")
        (tmp_path / "fort.51").write_bytes(b"  4\n")
        assert rt.read_formatted(51, "(I3)") == [4]

    def test_rewind_of_an_unconnected_unit_connects_it(self, tmp_path):
        rt.rewind_unit(52)
        assert (tmp_path / "fort.52").exists()

    def test_open_without_a_name_uses_fort_n(self, tmp_path):
        rt.open_unit(53)
        rt.write_formatted(53, "(A)", "z")
        rt.close_unit(53)
        assert (tmp_path / "fort.53").read_bytes() == b"z\n"


class TestInternalFiles:
    def test_round_trip(self):
        text = rt.write_internal("(I4,F7.3)", 42, 2.5)
        assert text == "  42  2.500"
        assert rt.read_internal(text, "(I4,F7.3)") == [42, 2.5]

    def test_record_longer_than_the_format_is_cut(self):
        assert rt.read_internal("12345", "(I2)") == [12]


class TestStandardStreamsAtCallTime:
    def test_stdout_replacement_between_calls(self, monkeypatch):
        first, second = io.StringIO(), io.StringIO()
        monkeypatch.setattr("sys.stdout", first)
        rt.write_formatted(6, "(A)", "one")
        monkeypatch.setattr("sys.stdout", second)
        rt.write_formatted(6, "(A)", "two")
        assert first.getvalue() == "one\n"
        assert second.getvalue() == "two\n"

    def test_stderr_replacement(self, monkeypatch):
        buffer = io.StringIO()
        monkeypatch.setattr("sys.stderr", buffer)
        rt.write_formatted(0, "(A)", "err")
        assert buffer.getvalue() == "err\n"


class TestThreadSafety:
    def test_concurrent_writers_with_their_own_units(self, tmp_path):
        threads_n, records = 8, 60
        errors: list[BaseException] = []
        barrier = threading.Barrier(threads_n)

        def work(k: int) -> None:
            try:
                unit = 100 + k
                barrier.wait()
                rt.open_unit(unit, tmp_path / f"w{k}.txt", status="replace")
                for i in range(records):
                    rt.write_formatted(unit, "(I3,1X,I5)", k, i)
                rt.close_unit(unit)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        for k in range(threads_n):
            content = lines(tmp_path / f"w{k}.txt")
            assert content[-1] == ""
            assert content[:-1] == [f"{k:3d} {i:5d}" for i in range(records)]

    def test_concurrent_writers_sharing_a_unit_never_interleave_records(self, tmp_path):
        threads_n, records = 8, 80
        path = tmp_path / "shared.txt"
        rt.open_unit(200, path, status="replace")
        errors: list[BaseException] = []
        barrier = threading.Barrier(threads_n)

        def work(k: int) -> None:
            try:
                barrier.wait()
                for i in range(records):
                    rt.write_formatted(200, "(2(I3,1X),A)", k, i, "x" * 20)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        rt.close_unit(200)
        assert errors == []
        content = lines(path)[:-1]
        assert len(content) == threads_n * records
        seen = set()
        for record in content:
            assert len(record) == 3 + 1 + 3 + 1 + 20 and record.endswith("x" * 20)
            seen.add((int(record[:3]), int(record[4:7])))
        assert seen == {(k, i) for k in range(threads_n) for i in range(records)}

    def test_concurrent_open_and_close_of_the_same_unit_numbers(self, tmp_path):
        threads_n, rounds = 6, 40
        errors: list[BaseException] = []
        barrier = threading.Barrier(threads_n)

        def work(k: int) -> None:
            try:
                barrier.wait()
                for r in range(rounds):
                    unit = 300 + (r % 3)
                    # Every thread opens the same three scratch units; the table must stay
                    # consistent no matter how the calls interleave.
                    rt.open_unit(unit, status="scratch")
                    try:
                        rt.write_formatted(unit, "(I1)", k % 10)
                    except OSError:
                        pass  # another thread closed the unit between our calls: allowed
                    rt.close_unit(unit)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        rt.close_all_units()
        assert rt._UNITS == {}

    def test_concurrent_reads_of_one_unit_hand_out_each_record_once(self, tmp_path):
        total = 200
        path = tmp_path / "records.txt"
        path.write_bytes("".join(f"{i:4d}\n" for i in range(total)).encode())
        rt.open_unit(400, path, status="old")
        taken: list[int] = []
        taken_lock = threading.Lock()
        errors: list[BaseException] = []

        def work() -> None:
            try:
                while True:
                    try:
                        (value,) = rt.read_formatted(400, "(I4)")
                    except EOFError:
                        return
                    with taken_lock:
                        taken.append(value)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert sorted(taken) == list(range(total))

    def test_concurrent_formatting_uses_independent_state(self):
        errors: list[BaseException] = []

        def work(k: int) -> None:
            try:
                for i in range(200):
                    expected = f"{k * 1000 + i:8d}"
                    assert rt.format_record("(I8)", [k * 1000 + i]) == expected
                    assert rt.write_internal("(F10.3)", float(i)) == f"{float(i):10.3f}"
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
