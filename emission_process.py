import numpy as np
import xarray as xr
import xesmf as xe


class GlobalEmissionProcessor:

    def __init__(self, camsds, ds_eagrid,
                 k=10,
                 sector="All_sources",
                 source="NOx"):

        self.camsds = camsds.squeeze()
        self.ds = ds_eagrid
        self.k = k
        self.sector = sector
        self.source = source

        # grids
        self.cams_lon = self.camsds.lon.values
        self.cams_lat = self.camsds.lat.values

        self.eagrid_lon = None
        self.eagrid_lat = None

        # data
        self.cams_emission = None
        self.eagrid_emission = None

        # regridded
        self.cams_bili = None
        self.cams_conserv = None
        self.eagrid_fine = None


    # --------------------------------------------------
    # helper
    # --------------------------------------------------
    def centers_to_edges(self, arr):

        d = np.diff(arr)

        edges = np.empty(len(arr)+1)
        edges[1:-1] = arr[:-1] + d/2
        edges[0] = arr[0] - d[0]/2
        edges[-1] = arr[-1] + d[-1]/2

        return edges


    # --------------------------------------------------
    # build target fine grid
    # --------------------------------------------------
    def build_target_grid(self):

        # CAMS resolution
        dlon = abs(self.cams_lon[1] - self.cams_lon[0])
        dlat = abs(self.cams_lat[1] - self.cams_lat[0])

        fine_dlon = round(dlon / self.k, 5)
        fine_dlat = round(dlat / self.k, 5)

        print(fine_dlon)
        print(fine_dlat)
        # --------------------------------
        # EAGrid spatial range
        # --------------------------------
        lon_min_ea = self.eagrid_lon_src.min()
        lon_max_ea = self.eagrid_lon_src.max()

        lat_min_ea = self.eagrid_lat_src.min()
        lat_max_ea = self.eagrid_lat_src.max()

        # --------------------------------
        # Find nearest CAMS grid indices
        # --------------------------------
        ix_min = np.argmin(np.abs(self.cams_lon - lon_min_ea))
        ix_max = np.argmin(np.abs(self.cams_lon - lon_max_ea))

        iy_min = np.argmin(np.abs(self.cams_lat - lat_min_ea))
        iy_max = np.argmin(np.abs(self.cams_lat - lat_max_ea))

        # --------------------------------
        # Expand to CAMS grid edges
        # --------------------------------
        lon_min = self.cams_lon[ix_min] - dlon/2
        lon_max = self.cams_lon[ix_max] + dlon/2

        lat_min = self.cams_lat[iy_min] - dlat/2
        lat_max = self.cams_lat[iy_max] + dlat/2

        
        # --------------------------------
        # Build fine grid
        # --------------------------------
        nx = int(round((lon_max - lon_min) / fine_dlon))
        ny = int(round((lat_max - lat_min) / fine_dlat))

        self.eagrid_lon = np.round(
            np.linspace(lon_min + fine_dlon/2,
                        lon_max - fine_dlon/2,
                        nx),
            4
        )

        self.eagrid_lat = np.round(
            np.linspace(lat_min + fine_dlat/2,
                        lat_max - fine_dlat/2,
                        ny),
            4
        )

    # --------------------------------------------------
    # read EAGrid
    # --------------------------------------------------
    def load_eagrid(self):

        sec_idx = int(self.ds.sector.to_index().get_loc(self.sector))
        src_idx = int(self.ds.source.to_index().get_loc(self.source))

        lat = self.ds["lat"].values
        lon = self.ds["lon"].values
        emission = self.ds["Emission"][sec_idx, src_idx, :].values

        lat_unique = np.unique(lat)
        lon_unique = np.unique(lon)

        ny = len(lat_unique)
        nx = len(lon_unique)

        grid = np.full((ny, nx), np.nan)

        for lo, la, val in zip(lon, lat, emission):

            i = np.searchsorted(lat_unique, la)
            j = np.searchsorted(lon_unique, lo)

            grid[i, j] = val

        self.eagrid_lat_src = lat_unique
        self.eagrid_lon_src = lon_unique
        self.eagrid_emission = np.nan_to_num(grid)


    # --------------------------------------------------
    # read CAMS
    # --------------------------------------------------
    def load_cams(self):

        self.cams_emission = (
            1e9 * 1e-2 * self.camsds["sum"].values
        )


    # --------------------------------------------------
    # regrid both datasets
    # --------------------------------------------------
    def regrid_all(self):

        grid_out = {
            "lon": self.eagrid_lon,
            "lat": self.eagrid_lat,
            "lon_b": self.centers_to_edges(self.eagrid_lon),
            "lat_b": self.centers_to_edges(self.eagrid_lat)
        }

        # CAMS grid
        grid_in_cams = {
            "lon": self.cams_lon,
            "lat": self.cams_lat,
            "lon_b": self.centers_to_edges(self.cams_lon),
            "lat_b": self.centers_to_edges(self.cams_lat)
        }

        # EAGrid grid
        grid_in_ea = {
            "lon": self.eagrid_lon_src,
            "lat": self.eagrid_lat_src,
            "lon_b": self.centers_to_edges(self.eagrid_lon_src),
            "lat_b": self.centers_to_edges(self.eagrid_lat_src)
        }

        # CAMS conservative
        cams_regridder_conserv = xe.Regridder(
            grid_in_cams,
            grid_out,
            "conservative"
        )
        cams_regridder_bili = xe.Regridder(
            grid_in_cams,
            grid_out,
            "bilinear"
        )

        self.cams_conserv = cams_regridder_conserv(self.cams_emission)
        self.cams_bili = cams_regridder_bili(self.cams_emission)

        # EAGrid conservative
        ea_regridder = xe.Regridder(
            grid_in_ea,
            grid_out,
            "conservative"
        )

        self.eagrid_fine = ea_regridder(self.eagrid_emission)


    # --------------------------------------------------
    # conservation test
    # --------------------------------------------------
    def check_conservation(self):

        # resolution
        dlon_src = abs(self.eagrid_lon_src[1] - self.eagrid_lon_src[0])
        dlat_src = abs(self.eagrid_lat_src[1] - self.eagrid_lat_src[0])
    
        dlon_dst = abs(self.eagrid_lon[1] - self.eagrid_lon[0])
        dlat_dst = abs(self.eagrid_lat[1] - self.eagrid_lat[0])
    
        # latitude arrays
        lat_src = self.eagrid_lat_src
        lat_dst = self.eagrid_lat
    
        # cell areas
        area_src_lat = self.gridcell_area(lat_src, dlon_src, dlat_src)
        area_dst_lat = self.gridcell_area(lat_dst, dlon_dst, dlat_dst)
    
        area_src = np.repeat(area_src_lat[:, None], len(self.eagrid_lon_src), axis=1)
        area_dst = np.repeat(area_dst_lat[:, None], len(self.eagrid_lon), axis=1)
    
        before = np.sum(self.eagrid_emission * area_src)
        after  = np.sum(self.eagrid_fine * area_dst)
    
        print("before:", before)
        print("after :", after)
    
        print("relative error:", abs(after-before)/before)
    # ------helper function of check conservation--------
    def gridcell_area(self, lat, dlon, dlat):

        R = 6371.0  # Earth radius km
    
        lat_rad = np.radians(lat)
        dlat_rad = np.radians(dlat)
        dlon_rad = np.radians(dlon)
    
        area = (R**2) * dlon_rad * (
            np.sin(lat_rad + dlat_rad/2) -
            np.sin(lat_rad - dlat_rad/2)
        )

        return area

    # --------------------------------------------------
    # cut windows
    # --------------------------------------------------
    def extract_window(self, lon_center, lat_center,
                       cams_pix=5):

        size = cams_pix * self.k
        half = size//2

        ix = np.argmin(abs(self.eagrid_lon - lon_center))
        iy = np.argmin(abs(self.eagrid_lat - lat_center))

        return (
        self.cams_bili[iy-half:iy+half, ix-half:ix+half],
        self.cams_conserv[iy-half:iy+half, ix-half:ix+half],
        self.eagrid_fine[iy-half:iy+half, ix-half:ix+half]
    )


    # --------------------------------------------------
    # filter invalid windows
    # --------------------------------------------------
    def valid_window(self, window,
                     threshold_sum=100,
                     threshold_pixels=1750):

        if np.nansum(window) < threshold_sum:
            return False

        if np.count_nonzero(window) < threshold_pixels:
            return False

        return True
    

if __name__ == "__main__":
    ds_eagrid = xr.open_dataset("Emission_grid_all.nc")
    cams_path = "CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v6.2_yearly_2010.nc"
    camsds = xr.open_dataset(cams_path)

    processor = GlobalEmissionProcessor(camsds, ds_eagrid)

    processor.load_cams()
    processor.load_eagrid()

    processor.build_target_grid()
    processor.regrid_all()

    processor.check_conservation()



