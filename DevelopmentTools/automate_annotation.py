# A set of functions to automatically annotate images for training a classification model of crosswalks.
# Requires a paid API key for the images (MAPBOX_API_KEY) but tile locations are free to access.

import overpy
from geopy.geocoders import Nominatim
import requests

MAPBOX_API_KEY = None  # Replace with your actual Mapbox API key
crosswalk_images_folder = "crosswalk_images"

def get_cross_walk_location(location_geo_code, bounds=0.001):
    """
    Retrieves a list of crosswalk locations (latitude, longitude) within a specified bounding box
    around a geocoded location string (e.g., 'Bloomsbury, London, UK') using the OpenStreetMap
    Overpass API.

    Args:
        location_geo_code (str):
            The place name to be geocoded (e.g., 'Bloomsbury, London, UK').
        bounds (float, optional):
            The distance in degrees (latitude/longitude) around the geocoded point
            to search for crosswalks, forming a square bounding box. Default is 0.001,
            which is roughly ~111 meters at the equator.

    Returns:
        list of tuple:
            A list of (latitude, longitude) pairs for crosswalks found in the bounding box.

    Raises:
        ValueError: If the geocoding process fails to find a location for the supplied geo_code.
    """
    # Initialize the Overpass API and the Nominatim geolocator
    over_api = overpy.Overpass()
    geo_locator = Nominatim(user_agent="crosswalk_locator")

    # Geocode the string location into latitude/longitude
    location = geo_locator.geocode(location_geo_code)
    if not location:
        raise ValueError("This location could not be geocoded - try an alternative name?")

    # Construct the bounding box around the location
    geo_bounding_box = (
        location.latitude - bounds,
        location.longitude - bounds,
        location.latitude + bounds,
        location.longitude + bounds
    )  # (South, West, North, East)

    # Overpass query to get highway=crossing nodes within the bounding box
    query = f"""
    [out:json][timeout:25];
    (
      node["highway"="crossing"]({geo_bounding_box[0]},{geo_bounding_box[1]},{geo_bounding_box[2]},{geo_bounding_box[3]});
    );
    out body;
    """

    # Execute the query and parse results
    result = over_api.query(query)

    # Extract crosswalk coordinates
    crosswalks = []
    for node in result.nodes:
        crosswalks.append((node.lat, node.lon))

    return crosswalks


def get_mapbox_aerial_image(location, output_file_name, zoom=18, size="256x256", style="satellite-v9",
                            api_key=MAPBOX_API_KEY):
    """
    Retrieves a static aerial image from the Mapbox Static Tiles API for a given (latitude, longitude) pair,
    saving the file locally. Useful for classification model training on crosswalk locations.

    Args:
        location (tuple):
            A 2-tuple of (longitude, latitude).
        output_file_name (str):
            The output file path where the image will be saved.
        zoom (int, optional):
            Zoom level for the Mapbox API, controlling scale of imagery. Defaults to 18.
        size (str, optional):
            The dimension of the requested tile (e.g., '256x256'). Defaults to '256x256'.
        style (str, optional):
            Mapbox style ID (e.g., 'satellite-v9'). Defaults to 'satellite-v9'.
        api_key (str, optional):
            Your Mapbox API access token. If None, will use the global MAPBOX_API_KEY.

    Returns:
        None

    Raises:
        HTTPError:
            If the request to the Mapbox API fails or returns a non-200 status code.
    """
    # Unpack location into longitude and latitude
    longitude, latitude = location

    # Construct the base URL for the map style
    base_url = f"https://api.mapbox.com/styles/v1/mapbox/{style}/static"
    # Format the center coordinates: latitude, longitude, zoom
    coords = f"{latitude},{longitude},{zoom}"
    # Build the full request URL for the static tile
    url = f"{base_url}/{coords}/{size}?access_token={api_key}"

    # Make the API request
    response = requests.get(url)

    # If the request succeeds, save the content to output_file_name
    if response.status_code == 200:
        with open(output_file_name, "wb") as file:
            file.write(response.content)
        print(f"Retrieved aerial image and saved it as {output_file_name}")
    else:
        print("Error, Error, Error!", response.status_code, response.text)


def get_crosswalk_images(geo_code, folder=crosswalk_images_folder, file_extension="png"):
    """
    Retrieves crosswalk images for a given geocoded location (e.g., 'Bloomsbury, London, UK'),
    using the Overpass API (for crosswalk node locations) and Mapbox Static Tiles API for aerial imagery.
    Saves each image into a specified folder.

    Args:
        geo_code (str):
            The location or area name to be geocoded (e.g., 'Bloomsbury, London, UK').
        folder (str, optional):
            The directory where downloaded images will be stored. Defaults to 'crosswalk_images'.
        file_extension (str, optional):
            The file extension for saved images (e.g., 'png'). Defaults to 'png'.

    Returns:
        list of str:
            A list of file names (as strings) corresponding to the crosswalk images saved locally.

    Raises:
        Exception:
            If a particular crosswalk image fails to download or save, the error is caught and printed.
            The function then continues to the next crosswalk location.
    """
    crosswalk_image_files = []
    # Get a list of crosswalk coordinates in the area specified by geo_code
    crosswalk_set = get_cross_walk_location(geo_code, bounds=0.1)

    for crosswalk in crosswalk_set:
        # Convert tuple items to float, then make it a list to create a unique file name
        formatted_crosswalk = list(map(float, crosswalk))
        stored_file_name = str(formatted_crosswalk)

        try:
            # Download aerial image from Mapbox at these coordinates
            get_mapbox_aerial_image(
                formatted_crosswalk,
                f"{folder}/{formatted_crosswalk}.{file_extension}"
            )
            crosswalk_image_files.append(stored_file_name)
        except Exception as e:
            # If an error occurs, print it and continue
            print(e)

    return crosswalk_image_files

# Example usage:
# print(len(get_cross_walk_location("Bloomsbury, London, UK")))
