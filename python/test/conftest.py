import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "cpu: indicate whether test is supported on cpu")
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
