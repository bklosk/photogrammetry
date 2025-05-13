#!/usr/bin/env python3

"""
2D Segmentation Script:
Crops a TIFF image based on a GeoJSON parcel and generates segmentation masks
using the Segment Anything Model (SAM).
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask # Renamed to avoid conflict with local 'mask' variables
import geopandas as gpd
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image # For saving masks

# --- Helper Functions ---

def load_geojson_parcel(geojson_path):
    """
    Loads GeoJSON parcel and returns a GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            print(f"ERROR: GeoJSON file {geojson_path} is empty or could not be read.")
            return None
        if gdf.crs is None:
            print(f"WARNING: GeoJSON file {geojson_path} has no CRS defined. Assuming EPSG:4326.")
            # Attempt to set a common default if CRS is missing, though this is a guess.
            # It's better if the GeoJSON has CRS info.
            try:
                gdf = gdf.set_crs("EPSG:4326", allow_override=False) # Common for GeoJSON
            except Exception as e:
                print(f"Could not set CRS for GeoJSON: {e}. Proceeding without explicit GeoJSON CRS.")
                # Fallback: proceed, hoping rasterio can handle or it matches raster by chance.
        
        print(f"Loaded GeoJSON from {geojson_path} with CRS: {gdf.crs}")
        return gdf
    except Exception as e:
        print(f"ERROR: Could not load GeoJSON {geojson_path}: {e}")
        return None

def crop_raster_to_parcel(raster_path, parcel_gdf):
    """
    Crops a raster image using a GeoJSON parcel geometry.
    Returns a NumPy array (H, W, C) in RGB uint8 format, or None on failure.
    """
    try:
        with rasterio.open(raster_path) as src:
            print(f"Opened raster {raster_path} with CRS: {src.crs}, Bands: {src.count}, Dtype: {src.meta['dtype']}")

            # Ensure parcel geometry is in the same CRS as the raster
            parcel_transformed_gdf = parcel_gdf
            if parcel_gdf.crs and src.crs and parcel_gdf.crs != src.crs:
                print(f"Reprojecting parcel from {parcel_gdf.crs} to raster CRS {src.crs}")
                parcel_transformed_gdf = parcel_gdf.to_crs(src.crs)
            
            shapes = [geom.__geo_interface__ for geom in parcel_transformed_gdf.geometry]

            try:
                out_image, out_transform = rio_mask(src, shapes, crop=True, all_touched=True, nodata=0)
            except ValueError as e:
                print(f"ERROR: Could not mask raster. This might be due to no overlap between parcel and raster: {e}")
                return None


            if out_image.size == 0:
                print("ERROR: Cropped image is empty (size 0). Check parcel/raster overlap and CRS.")
                return None

            # Process for SAM: expects HxWxC, RGB, uint8
            # out_image is (bands, height, width)
            num_bands_cropped = out_image.shape[0]
            print(f"Cropped image has {num_bands_cropped} bands. Shape: {out_image.shape}")

            if num_bands_cropped == 0:
                print("ERROR: Cropped image has 0 bands after masking.")
                return None

            # Select bands for RGB.
            if num_bands_cropped >= 3:
                # Assume first 3 bands are R,G,B or suitable for visual representation
                # For specific sensors (e.g., CIR with NIR,R,G), band order might need adjustment
                # For now, taking bands 0,1,2 as R,G,B respectively.
                rgb_image_bands = out_image[[0, 1, 2], :, :]
                print("Using first 3 bands for RGB.")
            elif num_bands_cropped == 1:
                print("Grayscale image, replicating to 3 channels for RGB.")
                rgb_image_bands = np.stack([out_image[0, :, :]] * 3, axis=0)
            else: # num_bands_cropped == 2
                print(f"WARNING: Image has {num_bands_cropped} bands. Using first for R, second for G, B will be zero.")
                band1 = out_image[0,:,:]
                band2 = out_image[1,:,:]
                zeros_ch = np.zeros_like(band1)
                rgb_image_bands = np.stack([band1, band2, zeros_ch], axis=0)
            
            # Transpose from (C, H, W) to (H, W, C)
            rgb_image_hwc = np.moveaxis(rgb_image_bands, 0, -1)

            if rgb_image_hwc.shape[0] == 0 or rgb_image_hwc.shape[1] == 0:
                print("ERROR: Cropped image has zero height or width after band processing.")
                return None

            # Normalize/scale to 0-255 uint8 using percentile stretching for robustness
            scaled_image_hwc = np.zeros_like(rgb_image_hwc, dtype=np.float32)

            for i in range(rgb_image_hwc.shape[2]): # For each channel (R, G, B)
                channel_data = rgb_image_hwc[:, :, i].astype(np.float32)
                
                # Handle channels that might be all zeros or constant after masking
                if np.all(channel_data == 0) or channel_data.min() == channel_data.max():
                    # If channel is flat (e.g. all nodata), keep it as 0
                    scaled_channel_data = np.zeros_like(channel_data, dtype=np.float32)
                else:
                    ch_min = np.percentile(channel_data, 2)
                    ch_max = np.percentile(channel_data, 98)
                    
                    if ch_max == ch_min: # Should be caught by above, but as a fallback
                        scaled_channel_data = np.zeros_like(channel_data, dtype=np.float32)
                    else:
                        scaled_channel_data = (channel_data - ch_min) / (ch_max - ch_min)
                
                scaled_image_hwc[:, :, i] = np.clip(scaled_channel_data, 0, 1) * 255.0

            img_uint8 = scaled_image_hwc.astype(np.uint8)
            print(f"Processed image for SAM. Shape: {img_uint8.shape}, dtype: {img_uint8.dtype}")
            return img_uint8

    except ImportError as e:
        print(f"ERROR: Import error: {e}. Make sure rasterio, geopandas are installed.")
        return None
    except Exception as e:
        print(f"ERROR: Failed during raster cropping or processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_and_save_sam_masks(image_rgb_uint8, model_type, checkpoint_path, output_dir):
    """
    Generates masks using SAM and saves them.
    """
    try:
        print(f"Loading SAM model: {model_type} from {checkpoint_path}")
        # SAM will attempt to use CUDA if available, otherwise CPU.
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        mask_generator = SamAutomaticMaskGenerator(sam)

        print("Generating masks with SAM (this may take a while)...")
        masks_sam = mask_generator.generate(image_rgb_uint8) # image_rgb_uint8 should be HxWxC, RGB, uint8

        if not masks_sam:
            print("SAM did not generate any masks.")
            return

        print(f"Generated {len(masks_sam)} masks.")
        os.makedirs(output_dir, exist_ok=True)

        for i, mask_data in enumerate(masks_sam):
            # mask_data['segmentation'] is a boolean HxW array
            mask_bool_array = mask_data['segmentation']
            mask_saveable_img = (mask_bool_array * 255).astype(np.uint8) # Convert boolean to 0/255 image
            
            pil_img = Image.fromarray(mask_saveable_img)
            mask_filename = os.path.join(output_dir, f"mask_{i+1:04d}.png")
            pil_img.save(mask_filename)
        
        print(f"All masks saved in {output_dir}")

    except FileNotFoundError:
        print(f"ERROR: SAM checkpoint file not found at {checkpoint_path}")
    except KeyError:
        print(f"ERROR: SAM model type '{model_type}' not recognized. Check available types.")
    except Exception as e:
        print(f"ERROR: Failed during SAM mask generation or saving: {e}")
        import traceback
        traceback.print_exc()

# --- Main Function ---
def main():
    # Hardcoded paths
    tif_image_path = "/workspaces/photogrammetry/data/boulder_flyover/images/n2w235.tif"
    geojson_parcel_path = "/workspaces/photogrammetry/data/boulder_flyover/metadata/parcel.geojson"
    sam_model_type = "vit_b"  # Example model type
    # IMPORTANT: Update this path to your actual SAM checkpoint file
    sam_checkpoint_path = "/workspaces/photogrammetry/models/sam_vit_b_01ec64.pth" 
    output_dir_path = "/workspaces/photogrammetry/output_sam_masks"

    print("Starting 2D segmentation process...")

    # 1. Load GeoJSON parcel
    print(f"Loading parcel from: {geojson_parcel_path}")
    parcel_gdf = load_geojson_parcel(geojson_parcel_path)
    if parcel_gdf is None:
        print("Exiting due to error loading GeoJSON.")
        return 1

    # 2. Crop raster to parcel
    print(f"Cropping raster {tif_image_path} using the parcel.")
    cropped_image_for_sam = crop_raster_to_parcel(tif_image_path, parcel_gdf)
    if cropped_image_for_sam is None:
        print("Exiting due to error cropping or processing raster.")
        return 1
    
    if cropped_image_for_sam.shape[0] == 0 or cropped_image_for_sam.shape[1] == 0:
        print("ERROR: Cropped image has zero height or width. Cannot proceed with SAM.")
        return 1

    # 3. Generate SAM masks
    print("Proceeding to SAM mask generation.")
    generate_and_save_sam_masks(
        cropped_image_for_sam, 
        sam_model_type, 
        sam_checkpoint_path, 
        output_dir_path
    )

    print("2D segmentation process finished.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
