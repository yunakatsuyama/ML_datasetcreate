import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from emission_process import GlobalEmissionProcessor



def make_testdata(camsds, ds_eagrid, cams_pix=5, sector="All_sources", source="NOx"):

    processor = GlobalEmissionProcessor(camsds, ds_eagrid)
    
    processor.load_cams()
    processor.load_eagrid()
    
    processor.build_target_grid()
    processor.regrid_all()
    
    processor.check_conservation()
    print("Global EAGrid regrid finished")

    # -----------------------------
    # Create Array for the cutting loop
    # -----------------------------
    # min_lon = processor.eagrid_lon_src.min()
    # max_lon = processor.eagrid_lon_src.max()
    # min_lat = processor.eagrid_lat_src.min()
    # max_lat = processor.eagrid_lat_src.max()

    # southern boundary of Hokkaido
    # min_lat = 41.35

    min_lon = 130.
    max_lon = 150.
    min_lat = 25.
    max_lat =  46.

    lon_array = np.arange(min_lon, max_lon, 0.1)
    lat_array = np.arange(min_lat, max_lat, 0.1)

    

    # ---------------------------------
    # Find valid center pixels
    # ---------------------------------
    eagrid_dataset = []  # [num of dataset, 50, 50]
    cams_conserv_dataset = [] # [num of dataset, 50, 50]
    cams_bili_dataset = [] # [num of dataset, 50, 50]
    center_lon = []  # [num of dataset]
    center_lat = [] # [num of dataset]
    for lon in lon_array:
        for lat in lat_array:
            
            cams_bili, cams_conserv, eagrid = processor.extract_window(lon, lat)
            
            # ---------------------------------------------------------
            # Check if the window has emission (delete sea, mountains)
            # ---------------------------------------------------------
            if processor.valid_window(eagrid):
                cams_bili_dataset.append(cams_bili)
                cams_conserv_dataset.append(cams_conserv)
                eagrid_dataset.append(eagrid)
                center_lon.append(lon)
                center_lat.append(lat)
                

    # ---------------------------------
    # Convert to numpy arrays
    # ---------------------------------

    center_lon = np.array(center_lon)
    center_lat = np.array(center_lat)
    cams_bili_dataset = np.array(cams_bili_dataset)
    cams_conserv_dataset = np.array(cams_conserv_dataset)
    eagrid_dataset = np.array(eagrid_dataset)


    print(f"center_lon : {center_lon}")
    # ---------------------------------
    # Save dataset
    # ---------------------------------

    # np.save("created_data/center_lon_test.npy", center_lon)
    # np.save("created_data/center_lat_test.npy", center_lat)
    # np.save("created_data/cams_bili_test.npy", cams_bili_dataset)
    # np.save("created_data/cams_conserv_test.npy", cams_conserv_dataset)
    # np.save("created_data/eagrid_dataset_test.npy", eagrid_dataset)

    

# ====================================================
# helper function
# calculate are 
# ====================================================
def gridcell_area(lat, dlon, dlat):
    R = 6371.0  # Earth radius km
    lat_rad = np.radians(lat)

    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)

    area = (R**2) * dlon_rad * (
        np.sin(lat_rad + dlat_rad/2) - np.sin(lat_rad - dlat_rad/2)
    )

    return area



# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":

    ds_eagrid = xr.open_dataset("Emission_grid_all.nc")
    #ds_eagrid = xr.open_dataset('/home/yuna/data_create/Emission_grid_all.nc')
    cams_path = "CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_nox_v6.2_yearly_2010.nc"
    camsds = xr.open_dataset(cams_path)

    make_testdata(camsds, ds_eagrid)
