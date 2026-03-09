def test_public_api_imports():
    from ssdiff import SSD

    # Sanity references to avoid “imported but unused”
    assert SSD is not None
