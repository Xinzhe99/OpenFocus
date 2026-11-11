"""Crash-isolated regression for control text event injection."""
import os
import subprocess
import sys

import pytest


def test_text_input_in_subprocess():
    pytest.importorskip('PyQt6')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    result = subprocess.run([sys.executable, __file__, 'exercise'], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _exercise():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import tempfile
    from core.control.protocol import encode
    from core.control.server import ControlServer
    from PyQt6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit
    from ui.control_surface import send_key

    app = QApplication([])
    edit = QPlainTextEdit()
    edit.show()
    for separator in ('\n', '\r\n', '\r'):
        edit.clear()
        send_key(edit, text='first' + separator + 'second')
        assert edit.toPlainText() == 'first\nsecond'
    edit.setPlainText('before selected after')
    cursor = edit.textCursor()
    cursor.setPosition(7)
    cursor.setPosition(15, cursor.MoveMode.KeepAnchor)
    edit.setTextCursor(cursor)
    send_key(edit, text='中😀\nnext')
    assert edit.toPlainText() == 'before 中😀\nnext after'
    for char in ('\x00', '\x01', '\x7f'):
        cursor = edit.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        edit.setTextCursor(cursor)
        old_text, old_selection = edit.toPlainText(), edit.textCursor().selectedText()
        try:
            send_key(edit, text='private-prefix' + char + 'private-suffix')
        except ValueError as exc:
            assert 'private' not in str(exc)
        else:
            raise AssertionError('unsupported control accepted')
        assert edit.toPlainText() == old_text
        assert edit.textCursor().selectedText() == old_selection
    with tempfile.TemporaryDirectory() as temporary:
        os.environ['OPENFOCUS_CONTROL_FILE'] = os.path.join(temporary, 'control.json')
        edit.current_fusion_method = lambda: 'none'
        server = ControlServer(edit, socket_name='OpenFocus-text-' + str(os.getpid()))
        def call(command, **args):
            return server.handle_line(encode({'id': '1', 'cmd': command, 'token': server.token,
                                               'args': args}), token=server.token)
        try:
            for command in ('ui.type', 'ui.key'):
                for separator in ('\n', '\r\n', '\r'):
                    edit.clear()
                    response = call(command, path='main', text='first' + separator + 'second')
                    assert response['ok'] and edit.toPlainText() == 'first\nsecond'
                    assert call('ui.context')['ok']
                edit.clear()
                assert call(command, path='main', text='a\n', **({'submit': True} if command == 'ui.type' else {}))['ok']
                assert edit.toPlainText() == ('a\n\n' if command == 'ui.type' else 'a\n')
                for char in ('\x00', '\x01', '\x7f'):
                    edit.setPlainText('original')
                    cursor = edit.textCursor()
                    cursor.select(cursor.SelectionType.Document)
                    edit.setTextCursor(cursor)
                    response = call(command, path='main', text='private' + char + 'secret')
                    assert response['error']['code'] == 'invalid_argument'
                    assert 'private' not in response['error']['message']
                    assert edit.toPlainText() == 'original'
                    assert edit.textCursor().selectedText() == 'original'
                    assert call('ui.context')['ok']
        finally:
            server.close()
    line = QLineEdit()
    returns = []
    line.returnPressed.connect(lambda: returns.append(True))
    send_key(line, text='a\r\nb')
    assert line.text() == 'ab' and len(returns) == 1
    edit.close()
    line.close()
    app.processEvents()


if __name__ == '__main__' and sys.argv[1:] == ['exercise']:
    _exercise()
