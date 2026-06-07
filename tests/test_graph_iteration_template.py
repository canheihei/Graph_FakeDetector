from pathlib import Path


TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "templates"
    / "graph-iteration.html"
)


def test_graph_iteration_template_supports_image_and_folder_upload():
    html = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'id="imageUpload"' in html
    assert 'accept="image/*"' in html
    assert 'id="folderUpload"' in html
    assert 'webkitdirectory' in html
    assert 'id="selectedImagePreview"' in html
