import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


src = rasterio.open('data/data/vv_vh_vv-vh.tif')
stacked_image = src.read()
profile = src.profile
affine = src.transform

training_shapefile = gpd.read_file('LULC/train_point.shp')
training_shapefile = training_shapefile.dropna()
if training_shapefile.crs != profile['crs']:
    training_shapefile = training_shapefile.to_crs(profile['crs'])

features = []
labels = []

label_mapping = {'agri': 1, 'vegetation': 2, 'builtup': 3, 'water': 4, 'barren': 5}

for _, row in training_shapefile.iterrows():
    geom = [row['geometry']]
    class_label = row['label']
    encoded_label = label_mapping[class_label]
    out_image, out_transform = mask(src, geom, crop=True)
    pixels = out_image.reshape(out_image.shape[0], -1).T
    valid_pixels = pixels[~np.isnan(pixels).any(axis=1)]
    features.extend(valid_pixels)
    labels.extend([encoded_label] * len(valid_pixels))
features = np.array(features)
labels = np.array(labels)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
rf = RandomForestClassifier(min_samples_split = 5, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
# Save the model
joblib.dump(rf, 'random_forest_model.pkl')
stacked_reshaped = stacked_image.reshape(stacked_image.shape[0], -1).T
predicted_classes = rf.predict(stacked_reshaped)
classified_image = predicted_classes.reshape((stacked_image.shape[1], stacked_image.shape[2]))

profile.update(dtype=rasterio.uint8, count=1, compress = "deflate")

with rasterio.open('data/data/output/classified_image_400.tif', 'w', **profile) as dst:
    dst.write(classified_image.astype(rasterio.uint8), 1)