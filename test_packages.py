import importlib
import pytest

packages = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("Flask", "flask"),
    ("Flask-CORS", "flask_cors"),
    ("Werkzeug", "werkzeug"),
    ("pandas", "pandas"),
    ("Pillow", "PIL"),
    ("firebase-admin", "firebase_admin"),
]

@pytest.mark.parametrize("pkg_name, import_name", packages)
def test_package_import(pkg_name, import_name):
    try:
        importlib.import_module(import_name)
    except Exception as e:
        pytest.fail(f"❌ {pkg_name} not available or failed to import: {e}")
