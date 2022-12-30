#Running all cells through VSCode, using the Jupyter Notebooks Extension



#%%
#Import required modules and set up prerequisites
#Google Earth Engine
import ee
# Authenticate your Earth Engine Account through your browser
ee.Authenticate()
# Initialize the Earth Engine library
ee.Initialize()

#For visualization in Python that are similar to MATLAB
import matplotlib.pyplot as plt
#Allows scientific computing
import numpy as np
#Mathematical functions and algorithms built on top of numpy
from scipy.stats import norm, gamma, f, chi2
#Allows frontend and raw HTML display
import IPython.display as disp
#'Magic Command' that allows the display in Jupyter Notebooks
%matplotlib inline
#Allows the creation of Leaflet Maps
import folium
# Define a method for displaying Earth Engine image tiles to folium map
def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
    tiles = map_id_dict['tile_fetcher'].url_format,
    attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    name = name,
    overlay = True,
    control = True
  ).add_to(self)

# Add EE drawing method to folium
folium.Map.add_ee_layer = add_ee_layer



# %%
#JSON data from geojson.io that is a small subset of data from a Sentinel-1A image
#This is a Rectangle over the Frankfurt Airport
geoJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              8.534317016601562,
              50.021637833966786
            ],
            [
              8.530540466308594,
              49.99780882512238
            ],
            [
              8.564186096191406,
              50.00663576154257
            ],
            [
              8.578605651855469,
              50.019431940583104
            ],
            [
              8.534317016601562,
              50.021637833966786
            ]
          ]
        ]
      }
    }
  ]
}

#Geometry Coordinates
coords = geoJSON['features'][0]['geometry']['coordinates']
#Area of Interest
aoi = ee.Geometry.Polygon(coords)
#Grabs a random image over the Area of Interest in the interferometric wide swath mode
#Grab Decibel and Float data types from the return
ffa_db = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD') 
                       .filterBounds(aoi) 
                       .filterDate(ee.Date('2020-08-01'), ee.Date('2020-08-31')) 
                       .first() 
                        # Clip the return 
                       .clip(aoi))
ffa_fl = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') 
                       .filterBounds(aoi) 
                       .filterDate(ee.Date('2020-08-01'), ee.Date('2020-08-31')) 
                       .first() 
                        # Clip the return   
                       .clip(aoi))
# Confirmation for Developer that the image has been grabbed correctly
# Return should be: ['VV', 'VH', 'angle'] in this example
ffa_db.bandNames().getInfo()



#%%
#Display generated image from VV(Vertical-Transmit, Vertical-Receive radar Transmission)
url = ffa_db.select('VV').getThumbURL({'min': -20, 'max': 0})
disp.Image(url=url, width=800)



# %%
#Generate an interactive folium map
location = aoi.centroid().coordinates().getInfo()[::-1]

# Make an RGB color composite image (VV,VH,VV/VH)
#VV is Vertical-Transmit, Vertical-Receive Radar
#VH is Vertical-Transmit, Horizontal-Receive Radar
rgb = ee.Image.rgb(ffa_db.select('VV'),
                   ffa_db.select('VH'),
                   ffa_db.select('VV').divide(ffa_db.select('VH')))

# Create the map object
m = folium.Map(location=location, zoom_start=12)

# Add the S1 rgb composite to the map object
m.add_ee_layer(rgb, {'min': [-20, -20, 0], 'max': [0, 0, 2]}, 'FFA')

# Add a layer control panel to the map
m.add_child(folium.LayerControl())

# Display the map
display(m)



# %%
#Creating and displaying a Histogram based on the pixels in the image
aoi_sub = ee.Geometry.Polygon(coords)

hist = ffa_fl.select('VV').reduceRegion(
    ee.Reducer.fixedHistogram(0, 0.5, 500),aoi_sub).get('VV').getInfo()
mean = ffa_fl.select('VV').reduceRegion(
    ee.Reducer.mean(), aoi_sub).get('VV').getInfo()
variance = ffa_fl.select('VV').reduceRegion(
    ee.Reducer.variance(), aoi_sub).get('VV').getInfo()

# Normalize and display the Histogram with numpy and matplotlib
a = np.array(hist)
x = a[:, 0]                 # array of bucket edge positions
y = a[:, 1]/np.sum(a[:, 1]) # normalized array of bucket contents
plt.grid()
plt.plot(x, y, '.')
plt.show()



# %%
#This is a Gamma Probability Density Distribution(https://en.wikipedia.org/wiki/Gamma_distribution), and can be verified by plotting said distribution 
alpha = 5
beta = mean/alpha
plt.grid()
plt.plot(x, y, '.', label='data')
plt.plot(x, gamma.pdf(x, alpha, 0, beta)/1000, '-r', label='gamma')
plt.legend()
plt.show()



# %%
#This sums up independent variables to create normal distributions. This is the Central Limit Theorem (https://en.wikipedia.org/wiki/Central_limit_theorem)
def X(n):
    return np.sum(np.cos(4*np.pi*(np.random.rand(n)-0.5)))/np.sqrt(n/2)

n= 10000
Xs = [X(n) for i in range(10000)]
y, x = np.histogram(Xs, 100, range=[-5,5])
plt.plot(x[:-1], y/1000, 'b.', label='simulated data')
plt.plot(x, norm.pdf(x), '-r', label='normal distribution')
plt.grid()
plt.legend()
plt.show()



# %%
# Generates the Equivalent Number of Looks to see if our data is representative
# The Google Earth Engine ENL is 4.4, so a number similar to that is representative
print("Equivalent Number of Looks:", mean ** 2 / variance)


# %%
