#!/usr/bin/env python3
"""
NAIP Orthophoto Fetcher using USGS NAIPPlus Service

This module fetches NAIP (National Agriculture Imagery Program) orthophotos
for a given address using the USGS NAIPPlus ImageServer with clipped exports.
The bounding box is automatically sized to approximately one acre.

Usage:
    from get_orthophoto import NAIPFetcher

    fetcher = NAIPFetcher()
    output_path, metadata = fetcher.get_orthophoto_for_address("1250 Wildwood Road, Boulder, CO")
"""

import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, Optional, Tuple
import requests
from utils import GeocodeUtils, FileUtils


class NAIPFetcher:
    """Class to fetch NAIP orthophotos using USGS NAIPPlus ImageServer."""

    def __init__(self):
        """Initialize the NAIP fetcher and query service capabilities."""
        self.service_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer/exportImage"
        # Related service info endpoint used to discover limits and native pixel
        # size.  Defaults are safe fallbacks in case the request fails.
        self.service_info_url = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
        self.max_image_width = 4000
        self.max_image_height = 4000
        self.native_pixel_size = 1.0
        self._fetch_service_info()

    def _fetch_service_info(self) -> None:
        """Query the service info endpoint to populate limits."""
        try:
            resp = requests.get(f"{self.service_info_url}?f=json", timeout=10)
            resp.raise_for_status()
            info = resp.json()

            # Safely extract values, ensuring they are numeric
            max_width = info.get("maxImageWidth", self.max_image_width)
            max_height = info.get("maxImageHeight", self.max_image_height)
            pixel_x = info.get("pixelSizeX", self.native_pixel_size)
            pixel_y = info.get("pixelSizeY", self.native_pixel_size)

            # Only update if we got valid numeric values
            if isinstance(max_width, (int, float)) and max_width > 0:
                self.max_image_width = max_width
            if isinstance(max_height, (int, float)) and max_height > 0:
                self.max_image_height = max_height
            if isinstance(pixel_x, (int, float)) and isinstance(pixel_y, (int, float)):
                self.native_pixel_size = min(pixel_x, pixel_y)

            print(
                f"Service limits - width: {self.max_image_width}, height: {self.max_image_height}, pixel size: {self.native_pixel_size}"
            )
        except Exception as exc:  # Broad catch to avoid breaking the flow
            print(f"Warning: failed to fetch service info: {exc}")

    def _calculate_optimal_size(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        pixel_size: float | None = None,
    ) -> str:
        """Calculate an image size matching the native pixel resolution."""

        if pixel_size is None:
            pixel_size = self.native_pixel_size

        mid_lat = (min_lat + max_lat) / 2.0
        height_m = (max_lat - min_lat) * 111_000
        width_m = (max_lon - min_lon) * 111_000 * math.cos(math.radians(mid_lat))

        width_px = int(width_m / pixel_size)
        height_px = int(height_m / pixel_size)

        width_px = max(1, min(width_px, self.max_image_width))
        height_px = max(1, min(height_px, self.max_image_height))

        return f"{width_px},{height_px}"

    def calculate_acre_bbox(
        self, latitude: float, longitude: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate a bounding box of approximately 4 acres around the given coordinates.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        # 4 acres = 16,188 square meters (for better LiDAR point coverage)
        # For a square area: side length = sqrt(16188) ≈ 127.2 meters
        # Convert to degrees (rough approximation)

        # Latitude: 1 degree ≈ 111,000 meters
        lat_degrees = 127.2 / 111000

        # Longitude varies by latitude: 1 degree longitude = cos(lat) * 111,000 meters
        lon_degrees = 127.2 / (111000 * math.cos(math.radians(latitude)))

        # Create square bounding box
        min_lat = latitude - lat_degrees / 2
        max_lat = latitude + lat_degrees / 2
        min_lon = longitude - lon_degrees / 2
        max_lon = longitude + lon_degrees / 2

        print(f"Generated ~4 acre bounding box:")
        print(f"  Center: {latitude:.6f}, {longitude:.6f}")
        print(f"  Bounds: {min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f}")
        print(
            f"  Size: ~{lat_degrees*111000:.1f}m x {lon_degrees*111000*math.cos(math.radians(latitude)):.1f}m"
        )

        return min_lon, min_lat, max_lon, max_lat

    def export_image(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        output_path: str,
        image_size: Optional[str] = None,
    ) -> Dict:
        """
        Export a clipped NAIP image from the USGS NAIPPlus service with fallback sizing.

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            output_path: Path to save the exported image
            image_size: Optional image size as ``"width,height"``.  ``None`` or
                ``"auto"`` computes a size based on the service's native pixel
                resolution for the provided bounding box.

        Returns:
            Dictionary containing export metadata

        Raises:
            Exception: If export fails
        """
        bbox = ",".join(map(str, [min_lon, min_lat, max_lon, max_lat]))

        if image_size in (None, "auto"):
            requested_size = self._calculate_optimal_size(
                min_lon, min_lat, max_lon, max_lat
            )
        else:
            requested_size = image_size
        fallback_sizes = ["2048,2048", "1024,1024", "512,512"]

        sizes_to_try = [requested_size] + fallback_sizes

        # Remove duplicates while preserving order
        seen = set()
        unique_sizes = []
        for size in sizes_to_try:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)

        last_error = None

        for attempt, size in enumerate(unique_sizes):
            params = {
                "bbox": bbox,
                "bboxSR": 4326,  # WGS84 coordinate system
                "size": size,  # Output image size in pixels
                "imageSR": 4326,  # Output coordinate system
                "format": "tiff",  # Output format
                "f": "image",  # Response format
            }

            print(f"Exporting NAIP image (attempt {attempt + 1}/{len(unique_sizes)}):")
            print(f"  Bounding box: {bbox}")
            print(f"  Image size: {size}")
            print(f"  Output: {output_path}")

            try:
                # Make request to USGS NAIPPlus service
                response = requests.get(self.service_url, params=params, timeout=60)
                response.raise_for_status()

                file_size = len(response.content)

                # Check if response is actually an image by examining content
                # The service sometimes returns JSON errors with image content-type
                if file_size < 1000:  # Very small response is likely an error
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get(
                            "message", "Unknown service error"
                        )
                        if (
                            "size limit" in error_msg.lower()
                            and attempt < len(unique_sizes) - 1
                        ):
                            print(f"  Size {size} too large, trying smaller size...")
                            last_error = Exception(f"NAIP service error: {error_msg}")
                            continue
                        else:
                            raise Exception(f"NAIP service error: {error_msg}")
                    except json.JSONDecodeError:
                        pass  # Not JSON, continue with normal processing

                # Additional check: if content type suggests image but size is suspiciously small
                content_type = response.headers.get("content-type", "")
                if content_type.startswith("image/") and file_size < 1000:
                    if attempt < len(unique_sizes) - 1:
                        print(
                            f"  Received very small image ({file_size} bytes), trying smaller size..."
                        )
                        last_error = Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )
                        continue
                    else:
                        raise Exception(
                            f"Received very small image ({file_size} bytes), likely an error response"
                        )

                # Save the image
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)

                print(f"Successfully exported {file_size:,} bytes to: {output_path}")
                if size != requested_size:
                    print(
                        f"Note: Used fallback size {size} instead of requested {requested_size}"
                    )

                # Create metadata
                metadata = {
                    "service": "USGS NAIPPlus",
                    "service_url": self.service_url,
                    "bbox": {
                        "min_longitude": min_lon,
                        "min_latitude": min_lat,
                        "max_longitude": max_lon,
                        "max_latitude": max_lat,
                    },
                    "bbox_string": bbox,
                    "coordinate_system": "EPSG:4326",
                    "image_size": size,  # Use actual successful size
                    "requested_size": requested_size,
                    "format": "tiff",
                    "file_size_bytes": file_size,
                    "export_time": datetime.now().isoformat(),
                    "output_path": output_path,
                }

                return metadata

            except requests.exceptions.RequestException as e:
                last_error = Exception(f"Failed to export image from USGS service: {e}")
                if attempt < len(unique_sizes) - 1:
                    print(f"  Network error, trying smaller size...")
                    continue
                else:
                    raise last_error
            except Exception as e:
                last_error = e
                if attempt < len(unique_sizes) - 1 and (
                    "size limit" in str(e).lower() or "small image" in str(e).lower()
                ):
                    print(f"  Error with size {size}: {e}")
                    continue
                else:
                    raise e

        # If we get here, all attempts failed
        raise last_error or Exception("All image size attempts failed")

    def save_metadata(
        self, metadata: Dict, output_dir: str, filename: str = "naip_metadata.json"
    ):
        """
        Save metadata to a JSON file.

        Args:
            metadata: Metadata dictionary to save
            output_dir: Directory to save the file
            filename: Name of the output file
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to: {filepath}")
        return filepath

    def get_orthophoto_for_address(
        self,
        address: str,
        output_dir: str = "../data",
        image_size: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Get NAIP orthophoto for a given address using USGS NAIPPlus service.

        Args:
            address: Street address to geocode and find imagery for
            output_dir: Directory to save downloaded files
            image_size: Desired image size.  Pass ``None`` or ``"auto"`` to use
                the service's native resolution for the computed bounding box.

        Returns:
            Tuple of (output_path, metadata)

        Raises:
            ValueError: If address is empty
            Exception: If geocoding or image export fails
        """
        # Validate address is not empty
        if not address or not address.strip():
            raise ValueError("Address cannot be empty")

        # Geocode the address
        print(f"Processing address: {address}")
        geocoder = GeocodeUtils()
        latitude, longitude = geocoder.geocode_address(address)

        # Calculate acre-sized bounding box
        min_lon, min_lat, max_lon, max_lat = self.calculate_acre_bbox(
            latitude, longitude
        )

        # Generate safe filename
        safe_name = FileUtils.get_safe_filename(address)
        filename = f"naip_orthophoto_{safe_name}.tif"
        output_path = os.path.join(output_dir, filename)

        # Export the image
        metadata = self.export_image(
            min_lon, min_lat, max_lon, max_lat, output_path, image_size
        )

        # Verify the file was created and return the actual path
        if os.path.exists(output_path):
            actual_output_path = output_path
        else:
            # If the expected file doesn't exist, search for any TIFF files in the output directory
            # that might have been created by the service
            import glob

            tiff_files = glob.glob(os.path.join(output_dir, "*.tif")) + glob.glob(
                os.path.join(output_dir, "*.tiff")
            )

            if tiff_files:
                # Sort by modification time and take the most recent
                tiff_files.sort(key=os.path.getmtime, reverse=True)
                actual_output_path = tiff_files[0]
                print(
                    f"Note: Expected file {output_path} not found, using {actual_output_path}"
                )
            else:
                # Fallback to the expected path (will likely cause an error downstream but that's appropriate)
                actual_output_path = output_path

        # Add address info to metadata
        metadata["address"] = address
        metadata["geocoded_coordinates"] = {
            "latitude": latitude,
            "longitude": longitude,
        }
        metadata["actual_output_path"] = actual_output_path

        # Save metadata
        self.save_metadata(metadata, output_dir)

        return actual_output_path, metadata

    def _get_orthophoto_url(self, address: str) -> str:
        """
        Get the orthophoto URL for a given address.
        This is a helper method that can be mocked in tests.

        Args:
            address: Street address to get URL for

        Returns:
            URL string for the orthophoto
        """
        # This is a placeholder method that tests can mock
        # In reality, this would construct an appropriate URL based on the address
        return f"{self.service_url}?address={address}"


def get_orthophoto_for_address(
    address: str, output_dir: str = "../data", image_size: Optional[str] = None
) -> Tuple[str, Dict]:
    """
    Convenience function to get NAIP orthophoto for an address.

    Args:
        address: Street address to geocode and find imagery for
        output_dir: Directory to save downloaded files
        image_size: Desired image size.  ``None`` uses automatic calculation.

    Returns:
        Tuple of (output_path, metadata)
    """
    fetcher = NAIPFetcher()
    return fetcher.get_orthophoto_for_address(address, output_dir, image_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch NAIP orthophoto for a given address using USGS NAIPPlus service"
    )
    parser.add_argument(
        "address",
        help="Street address to geocode and find imagery for (e.g., '1250 Wildwood Road, Boulder, CO')",
    )
    parser.add_argument(
        "--output-dir",
        default="../data",
        help="Directory to save downloaded files (default: ../data)",
    )
    parser.add_argument(
        "--image-size",
        default=None,
        help=(
            "Image size as 'width,height'. Use 'auto' or omit to fetch at "
            "native resolution"
        ),
    )

    args = parser.parse_args()

    try:
        output_path, metadata = get_orthophoto_for_address(
            args.address, args.output_dir, args.image_size
        )
        print(f"\nSuccess!")
        print(f"Image saved to: {output_path}")
        print(f"File size: {metadata.get('file_size_bytes', 0):,} bytes")
        bbox = metadata.get("bbox", {})
        print(
            f"Bounding box: {bbox.get('min_longitude', 'N/A'):.6f}, {bbox.get('min_latitude', 'N/A'):.6f} to {bbox.get('max_longitude', 'N/A'):.6f}, {bbox.get('max_latitude', 'N/A'):.6f}"
        )
        print(f"Image size: {metadata.get('image_size', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
