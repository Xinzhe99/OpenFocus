"""Font indexing and style resolution with controlled font metadata."""
import pytest

from utils import fonts


@pytest.fixture(autouse=True)
def clear_font_state():
    fonts.clear_font_cache()
    yield
    fonts.clear_font_cache()


class Face:
    def __init__(self, family, style, size, covered=True):
        self.family, self.style, self.size, self.covered = family, style, size, covered

    def getname(self):
        return self.family, self.style

    def getmask(self, char):
        class Mask:
            size = (1, 1)

            def __bytes__(self):
                if char == '\ue000' or (char == '漢' and not self_face.covered):
                    return b'\x00'
                return b'\x02' if 'Bold' in self_face.style else b'\x01'
        self_face = self
        return Mask()


class FakeFonts:
    def __init__(self, metadata):
        self.metadata = metadata
        self.calls = []

    def truetype(self, path, size, index=0):
        self.calls.append((str(path), size, index))
        data = self.metadata.get((str(path), index))
        if data is None:
            raise OSError('unreadable face')
        return Face(*data[:2], size, *data[2:])

    def load_default(self):
        return Face('default', 'Regular', 12)


def setup_fonts(monkeypatch, root, metadata):
    fake = FakeFonts(metadata)
    monkeypatch.setattr(fonts, '_font_dirs', lambda: [str(root)])
    monkeypatch.setattr(fonts, '_pil_font', lambda: fake)
    monkeypatch.setattr(fonts, '_FALLBACK_CANDIDATES', ())
    fonts.clear_font_cache()
    return fake


def touch(root, name):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def test_recursive_index_styles_ttc_cache_and_clear(tmp_path, monkeypatch):
    regular = touch(tmp_path, 'truetype/family/Regular.ttf')
    collection = touch(tmp_path, 'opentype/family/Faces.ttc')
    touch(tmp_path, 'broken.ttf')
    touch(tmp_path, 'ignored.txt')
    try:
        (tmp_path / 'loop').symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pass
    fake = setup_fonts(monkeypatch, tmp_path, {
        (regular, 0): ('Example', 'Regular'),
        (collection, 0): ('Other', 'Regular'),
        (collection, 1): ('Example', 'Bold'),
    })
    normal, family = fonts.load_font(24, 'Example')
    bold, bold_family = fonts.load_font(24, 'Example', True)
    assert family == bold_family == 'Example'
    assert normal.getname()[1] == 'Regular'
    assert bold.getname()[1] == 'Bold'
    assert bytes(normal.getmask('A')) != bytes(bold.getmask('A'))
    assert (collection, 24, 1) in fake.calls
    count = len(fake.calls)
    assert fonts.load_font(24, 'Example') == (normal, family)
    assert fonts.load_font(24, 'Example', True) == (bold, bold_family)
    assert len(fake.calls) == count
    fonts.clear_font_cache()
    fonts.load_font(24, 'Example')
    assert len(fake.calls) > count


def test_budget_and_same_family_style_fallback(tmp_path, monkeypatch):
    regular = touch(tmp_path, 'a.ttf')
    touch(tmp_path, 'b.ttf')
    fake = setup_fonts(monkeypatch, tmp_path, {(regular, 0): ('Example', 'Regular')})
    monkeypatch.setattr(fonts, '_MAX_SCANNED_FONTS', 1)
    face, family = fonts.load_font(18, 'Example', True)
    assert family == 'Example' and face.getname()[1] == 'Regular'
    assert sum(size == 12 for _, size, _ in fake.calls) == 1


def test_corrupt_candidate_and_cjk_style_fallback(tmp_path, monkeypatch):
    bad = touch(tmp_path, 'a.ttf')
    regular = touch(tmp_path, 'b.ttf')
    cjk = touch(tmp_path, 'c.ttf')
    cjk_bold = touch(tmp_path, 'd.ttf')
    fake = setup_fonts(monkeypatch, tmp_path, {
        (bad, 0): ('Latin', 'Bold', False),
        (regular, 0): ('Latin', 'Regular', False),
        (cjk, 0): ('CJK', 'Regular', True),
        (cjk_bold, 0): ('CJK', 'Bold', True),
    })
    fonts._scan_family_index()
    del fake.metadata[(bad, 0)]
    monkeypatch.setattr(fonts, '_cjk_capable_candidates', lambda: (cjk,))
    face, family, substituted = fonts.load_font_for_text(20, 'Latin', '漢', True)
    assert (family, substituted, face.getname()[1]) == ('CJK', 'Latin', 'Bold')


def test_pillow_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(fonts, '_pil_font', lambda: None)
    fonts.clear_font_cache()
    assert fonts.load_font(12, 'Missing', True) == (None, '')
    assert fonts.load_font_for_text(12, 'Missing', '漢', True) == (None, '', '')
