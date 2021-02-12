class VisData(object):
    # Xarray Dataset
    ds = None

    def __init__(self, ds):
        self.ds = ds.copy()
