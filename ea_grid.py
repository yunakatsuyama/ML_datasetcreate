import numpy as np
import xesmf as xe


class EAGridGlobalRegridder:

    def __init__(self, ds_eagrid, k=10,
                 sector="All_sources",
                 source="NOx"):

        self.ds = ds_eagrid
        self.k = k
        self.sector = sector
        self.source = source

        self.lat_e = None
        self.lon_e = None
        self.emission_grid = None

        self.lat_fine = None
        self.lon_fine = None
        self.emission_fine = None


    # --------------------------------------------------
    # Build full EAGrid grid
    # --------------------------------------------------
    def build_eagrid_full(self):

        sec_idx = int(self.ds.sector.to_index().get_loc(self.sector))
        src_idx = int(self.ds.source.to_index().get_loc(self.source))

        lat = self.ds["lat"].values
        lon = self.ds["lon"].values
        emission = self.ds["Emission"][sec_idx, src_idx, :].values

        lat_unique = np.unique(lat)
        lon_unique = np.unique(lon)

        ny = len(lat_unique)
        nx = len(lon_unique)

        emission_grid = np.full((ny, nx), np.nan)

        for lo, la, val in zip(lon, lat, emission):

            i = np.searchsorted(lat_unique, la)
            j = np.searchsorted(lon_unique, lo)

            emission_grid[i, j] = val

        emission_grid = np.nan_to_num(emission_grid)

        self.lat_e = lat_unique
        self.lon_e = lon_unique
        self.emission_grid = emission_grid


    # --------------------------------------------------
    # Build global fine grid
    # --------------------------------------------------
    def build_fine_grid_global(self):

        dlat = np.abs(self.lat_e[1] - self.lat_e[0])
        dlon = np.abs(self.lon_e[1] - self.lon_e[0])

        fine_dlat = dlat / self.k
        fine_dlon = dlon / self.k

        lat_min = self.lat_e.min() - dlat / 2
        lat_max = self.lat_e.max() + dlat / 2

        lon_min = self.lon_e.min() - dlon / 2
        lon_max = self.lon_e.max() + dlon / 2

        self.lat_fine = np.arange(
            lat_min + fine_dlat / 2,
            lat_max,
            fine_dlat
        )

        self.lon_fine = np.arange(
            lon_min + fine_dlon / 2,
            lon_max,
            fine_dlon
        )


    # --------------------------------------------------
    # Helper
    # --------------------------------------------------
    def centers_to_edges(self, arr):

        d = np.diff(arr)

        edges = np.empty(len(arr) + 1)

        edges[1:-1] = arr[:-1] + d / 2
        edges[0] = arr[0] - d[0] / 2
        edges[-1] = arr[-1] + d[-1] / 2

        return edges


    # --------------------------------------------------
    # Global conservative regrid
    # --------------------------------------------------
    def regrid_eagrid_global(self):

        grid_in = {
            "lat": self.lat_e,
            "lon": self.lon_e,
            "lat_b": self.centers_to_edges(self.lat_e),
            "lon_b": self.centers_to_edges(self.lon_e)
        }

        grid_out = {
            "lat": self.lat_fine,
            "lon": self.lon_fine,
            "lat_b": self.centers_to_edges(self.lat_fine),
            "lon_b": self.centers_to_edges(self.lon_fine)
        }

        regridder = xe.Regridder(
            grid_in,
            grid_out,
            "conservative"
        )

        self.emission_fine = regridder(self.emission_grid)


    # --------------------------------------------------
    # Extract window after regrid
    # --------------------------------------------------
    def extract_window_fine(self, lon_center, lat_center,
                            cams_pix=5):

        size = cams_pix * self.k
        half = size // 2

        ix = np.argmin(np.abs(self.lon_fine - lon_center))
        iy = np.argmin(np.abs(self.lat_fine - lat_center))

        lon_slice = slice(ix-half, ix+half)
        lat_slice = slice(iy-half, iy+half)

        return self.emission_fine[lat_slice, lon_slice]