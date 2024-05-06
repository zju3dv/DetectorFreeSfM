
# Ray utils
def chunks(lst, n, length=None):
    """Yield successive n-sized chunks from lst."""
    try:
        _len = len(lst)
    except TypeError as _:
        assert length is not None
        _len = length

    for i in range(0, _len, n):
        yield lst[i : i + n]
    # TODO: Check that lst is fully iterated