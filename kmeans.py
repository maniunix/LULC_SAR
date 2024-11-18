import rasterio
import os
from rasterio.plot import reshape_as_image
from sklearn.cluster import KMeans

raster_loc = "vv_vh_vv-vh.tif"
src = rasterio.open(raster_loc)
stacked_image = src.read()
profile = src.profile

image_2d = reshape_as_image(stacked_image)
rows, cols, bands = image_2d.shape
image_flattened = image_2d.reshape((-1, bands))
num_clusters = 15

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(image_flattened)

clustered_image = clusters.reshape((rows, cols))
profile.update(dtype=rasterio.uint8, count=1)
file_name = os.path.basename(raster_loc).replace(".tif", "_kmeans_15.tif") 
with rasterio.open(file_name, "w", **profile) as dst:
    dst.write(clustered_image,1)