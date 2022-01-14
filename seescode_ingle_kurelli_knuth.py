'''
***Predicting NDVI Values from GLOBE Land Cover RGB Measurements with Multiple Linear Regression***
 > NASA SEES 2020 Mosquito Mapper Independent Student Project
 NOTE:: $NDVI = \frac {NIR - R}{NIR + R}$
'''

'''
################################################################################
# Section 1 -- Upload GLOBE Land Cover CSV file & Convert to pandas dataframe  #
# > NOTE: This csv was downloaded 8:48pm, July 20, 2020                        #
################################################################################
'''
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from google.colab import files

#Upload GLOBE Land Cover csv & save as pandas df
import pandas as pd
csv_uploaded = files.upload()
for fn in csv_uploaded.keys():
  print('User uploaded file "{name}"'.format(name=fn))
  globe_df = pd.read_csv("{name}".format(name=fn))

'''
################################################################################
#                       Section 2 -- Organize globe_df                         #
# > Organize by date, then user, then location                                 #
# > Remove unneccessary entries                                                #
################################################################################
'''

## **Step 1** -- Sort Values by Date, School Name, Site Name
globe_df = globe_df.sort_values(by=['Measured At','School Name','Site Name'])
globe_df = globe_df[['Measured At','School Name','Site Name','Latitude','Longitude','Elevation','Measured Value']]
globe_df.reset_index(drop=True, inplace=True)

## **Step 2** -- Find & remove duplicate entries
# > Note: Duplicate entries mean that ALL column values match for that particular row to a previous row.

### Find Duplicate Entries
duplicateRowsDF = globe_df[globe_df.duplicated()]
print(f"Duplicate Rows except first occurrence based on all columns are :\n{duplicateRowsDF}")

### Remove Duplicate and Null Entries
globe_df.drop_duplicates(keep='last',inplace=True,ignore_index=True)
globe_df.dropna(inplace=True)

## **Step 3** -- Remove entries that are NOT from the 2020 SEES group
# > Keep entry IF School Name is *United States of America Citizen Science*
indexNames = globe_df[ (globe_df['School Name'] != 'United States of America Citizen Science') ].index
globe_df.drop(indexNames , inplace=True)
globe_df.reset_index(drop=True, inplace=True)

##**Step 4** -- Remove any entry before Official SEES 2020 Start Date: Jun 1, 2020
times = globe_df['Measured At']
start = -1
for x in range(len(times)):
  if times[x][0:7] == "2020-06":
    start = x
    break

indexNames = globe_df[ (globe_df.index < start) ].index
globe_df.drop(indexNames , inplace=True)

globe_df.reset_index(drop=True, inplace=True)

'''
################################################################################
#            Section 3 -- Create and download csv_for_appeears                 #
# > For AppEEARS, columns should be: 'ID', 'Latitude', 'Longitude'             #
# NOTE: AppEEARS Layer = NDVI index & 250m pixel size                          #
# (This is the most focussed pixel size, which is still > than SEES 100m grid  #
################################################################################
'''
## **Step 1** -- Copy globe_df & remove extranneous columns (Keep only 'Latitude' and 'Longitude')
globe_df_for_appeears = globe_df[['Latitude', 'Longitude']]
globe_df_for_appeears

## **Step 2** -- Convert globe_df_for_appeears to csv & Create ID Column ('ID' = index)
globe_df_for_appeears.to_csv('csv_for_appeears',index_label='ID')

## **Step 3** -- Download csv and send data request to AppEEARS for satellite analysis
# SOURCE: AppEEARS = https://lpdaacsvc.cr.usgs.gov/appeears/
# Note: Need to split into 3 chunks, given AppEEARS entry limit of 1000 samples per request.
# Note: This downloads WITH column headings! So when inputting into AppEEARS, ignore the first line!
files.download('csv_for_appeears')

'''
################################################################################
#            Section 4 -- Extracting Median RGB Values from each Image         #
################################################################################
'''

## **Step 1** -- Compress image by converting to thumbnail
# > Note: Max thumbnail size is 100x100 pixels
def convert_to_thumbnail(img,shape=(100,100,3)):
  return img.thumbnail(shape)

##**Step 2** -- Import required packages
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
import cv2
from collections import Counter
import urllib.request
from PIL import Image, ImageEnhance

def get_img_from_url(url):
  '''*Extract image from url. Sharpen/brighten image.*
  Param: String url
  Output: nparray representation of image (sharpened & enhanced)
  '''
  urllib.request.urlretrieve(url, "sample.png")
  img = Image.open("sample.png")

  #SHARPEN IMAGE
  sharpener = ImageEnhance.Sharpness(img.convert('RGB'))
  sharpened = sharpener.enhance(2.0)

  #ENHANCE IMAGE BRIGHTNESS
  converter = ImageEnhance.Color(sharpened)
  converted = converter.enhance(1.5)
  return converted

def get_median_rgb_vals(image_as_array):
  '''*Return median RGB values from 1 image*
  Param: np array that represents an image
  Output: dictionary of pixel vals
  '''
  img = image_as_array
  height, width, channels = img.shape
  pixel_rgb = []
  for h in range(height):
    for w in range(width):
      pixel = img[h,w]
      pixel_row = {'pr':pixel[0],'pg':pixel[1],'pb':pixel[2]}
      pixel_rgb.append(pixel_row)
  pixel_rgb = pd.DataFrame(pixel_rgb)

  med_r = np.median(pixel_rgb['pr'])
  med_g = np.median(pixel_rgb['pg'])
  med_b = np.median(pixel_rgb['pb'])
  dict_row = {'r':med_r, 'g':med_g, 'b':med_b}
  return dict_row

## **Step 4** -- Either create `rgb_vals` or upload the csv if that process has been completed previously
# > Note: Generating this csv takes some time. So, if you already have the rgb_vals csv, just upload it using the appropriate code cells!
###If you do NOT have the `rgb_vals` csv, run the cells below to generate:
# * Dateframe `rgb_vals`
# * CSV `rgb_vals`
rgb_vals = []
i = 0

for url_img in globe_df['Measured Value']:
  #obtain img & compress
  img = get_img_from_url(url_img)
  convert_to_thumbnail(img)

  #convert img to np.array
  img_array = np.array(img)

  #get 3 calc veg indices val
  dict_row = get_median_rgb_vals(img_array)

  rgb_vals.append(dict_row)
  print('"{num}" -- "{veg}"'.format(num=i,veg=dict_row))
  i+=1

rgb_vals = pd.DataFrame(rgb_vals)
rgb_vals.reset_index(inplace=True)
rgb_vals.columns = ['ID','Median r','Median g','Median b']

#converts to csv
rgb_vals.to_csv('rgb_vals',index=False)

#downloads as csv
files.download('rgb_vals')

"""###If you DO have the `rgb_vals` CSV, run the cells below to upload and save as a Dataframe:"""
rgb_vals = files.upload()
for fn in rgb_vals.keys():
  print('User uploaded file "{name}"'.format(name=fn))
  rgb_vals = pd.read_csv("{name}".format(name=fn))

'''
#################################################################################
# Section 5 -- Upload AppEEARS csv chunks, Convert to pandas dataframe, & Merge #
#################################################################################
'''
appears_pt_1 = files.upload()
for fn in appears_pt_1.keys():
  print('User uploaded file "{name}"'.format(name=fn))
  appears_pt_1 = pd.read_csv("{name}".format(name=fn))

appears_pt_2 = files.upload()
for fn in appears_pt_2.keys():
  print('User uploaded file "{name}"'.format(name=fn))
  appears_pt_2 = pd.read_csv("{name}".format(name=fn))

appears_pt_3 = files.upload()
for fn in appears_pt_3.keys():
  print('User uploaded file "{name}"'.format(name=fn))
  appears_pt_3 = pd.read_csv("{name}".format(name=fn))

#merge dataframes
frames = [appears_pt_1, appears_pt_2, appears_pt_3]
appears_df = pd.concat(frames)
appears_df.reset_index(inplace=True, drop=True)

'''
#################################################################################
#                       Section 6 -- Organize appears_df                        #
#################################################################################
'''

##**Step 1** -- Remove all unnecessary columns
# > Keep: `ID`, `Latitude`, `Longitude`, `Date`, `MOD13A1_006__500m_16_days_NDVI`
appears_df = appears_df[['ID','Latitude','Longitude','Date','MOD13Q1_006__250m_16_days_NDVI']]

##**Step 2** -- Rename `MOD13A1_006__500m_16_days_NDVI` to instead be `NDVI`
appears_df.columns = ['ID','Latitude','Longitude','Date','NDVI']

## **Step 3** -- Clean-up NDVI column
# > For each location, the AppEEARS data request returned NDVI for 3 different dates: May 24, 2020, June 9, 2020, and June 25, 2020.
# > The AppEEARS data request also listed some NDVIs as -3000, an extraneous value that denotes the uselessness of that particular NDVI measurement.
# > Find median of all non-extranneous ndvis. If all are extranneous, record the corresponding ID
def calc_median_ndvi(ID):
  #Given a unique ID, extract the corresponding NDVI values from the three different dates.
  indexNames = appears_df[ (appears_df['ID'] == ID) ].index
  ids_ndvi = []
  count = 0 #num of extranneous vals

  # Iterate through each NDVI val
  for i in indexNames:
    ndvi_val = appears_df['NDVI'][i]
    if (ndvi_val < 0) or (ndvi_val > 1):
      count += 1
    else:
      ids_ndvi.append(ndvi_val)

  #If all 3 ndvis = extranneous, recorded that ID to ignore in future calculations.
  if count == 3:
    return -3000
  else:
    return np.median(ids_ndvi)

# For all ID's, calc median ndvi && append to df
df_as_list = []
for id in rgb_vals['ID']:
  r_val = rgb_vals['Median r'][id]
  g_val = rgb_vals['Median g'][id]
  b_val = rgb_vals['Median b'][id]
  ndvi_index = calc_median_ndvi(id)

  dict_row = {'ID':id, 'Med_r':r_val, 'Med_g':g_val,'Med_b':b_val,'NDVI':ndvi_index}
  df_as_list.append(dict_row)

df = pd.DataFrame(df_as_list)

# Drop rows w NDVI < -100
indexNames = df[ (df['NDVI'] < -100 ) ].index
df.drop(indexNames , inplace=True)
df.reset_index(drop=True, inplace=True)

# These ID's had no measured NDVI value, so we are excluding from test-train dataset!
print(f"Excluding the following ID's, since no valid NDVI:\n{indexNames}")


'''
################################################################################
#  Section 7 -- Determine relationship between GLOBE vegetation index & NDVI   #
# > x/features = RGB values
# > y/targets = NDVI
################################################################################
'''

##**Step 1** -- Import important packages
# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics

##**Step 2** -- Get Summary Statistics of RGB & NDVI
df_stats = df.describe()
df_stats = df_stats[['Med_r','Med_g','Med_b','NDVI']]

##**Step 3** -- Visualize RGB and NDVI in a 3D format
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ptx, pty, ptz = np.array(df['Med_r']), np.array(df['Med_g']), np.array(df['Med_b'])
values = np.array(df['NDVI'])
p = ax.scatter3D(ptx, pty, zs=ptz, c=values, cmap='viridis')

cbar = fig.colorbar(p, ax=ax)
cbar.ax.set_ylabel('NDVI', rotation=270, labelpad=10)
plt.title("RGB vs NDVI")
ax.set_xlabel("r")
ax.set_ylabel("g")
ax.set_zlabel("b")
plt.savefig("3D_rgb_v_ndvi.jpg")
files.download("3D_rgb_v_ndvi.jpg")
plt.show()

plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(df['NDVI'])

## **Step 4** -- Why Multiple LINEAR Regression Model?
# > Visualize the relationship between each feature and the response using scatterplots
# > No Scatterplot has stronger relationship with NDVI, so must be multiple LINEAR regressive model, rather than multiple POLYNOMIAL regressive model
# > NOTE: Shaded Region represents 95% Confidence Interval

f = sns.pairplot(df,x_vars=['Med_r','Med_g','Med_b'],y_vars='NDVI', height=7, aspect=0.7, kind='reg').fig

# Use meaningful titles for the columns
titles = ["r v NDVI", "g v NDVI", "b v NDVI"]
for ax, title in zip(f.axes, titles):
    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

# f.write_image(file='Pairplot of RGB Values vs NDVI', format='.png')
# files.download("Pairplot of RGB Values vs NDVI")
# plt.title("RGB Values vs NDVI")

"""##**Step 5** -- Multiple Linear Regression

>$\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$
* $\hat{y}$ is the predicted response
* $\beta_0$ is the intercept
* $\beta_1$ is the coefficient for $x_1$ (first feature)
* $\beta_n$ is the coefficient for $x_n$ (nth feature)

> In this case:
* $\widehat{NDVI} = \beta_0 + \beta_1 \times r + \beta_2 \times g + \beta_3 \times b$
"""

###**Step 5.A** -- Defining x/features & y/response
feature_names = ['Med_r','Med_g','Med_b']

x = df[['Med_r','Med_g','Med_b']].values
y = df['NDVI'].values

#OR Can do this:
#x = np.column_stack((df['Med_r'],df['Med_g'],df['Med_b']))
#y = df['NDVI'].values

###**Step 5.B** -- Determining optimal Random_State (Results in lowest RSME?)
rand_range = range(501)
rmse_scores = []
for k in rand_range:
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=k)
  linreg = LinearRegression()
  linreg.fit(x_train,y_train)
  y_pred = linreg.predict(x_test)
  val = (np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
  rmse_scores.append(val) #DONT FORGET GOTTA AVG
min = np.min(rmse_scores)
rand_st = np.argmin(rmse_scores)

plt.plot(rand_range, rmse_scores, 'r')
plt.xlabel('Random State')
plt.ylabel('RMSE Scores')
plt.title("Random State vs RMSE Scores")
plt.savefig("Random_State_vs_RMSE_Scores.png")
files.download("Random_State_vs_RMSE_Scores.png")
plt.show()

###**Step 5.C** -- Performing Multiple Linear Regression using Random_State
indices = np.array(df['ID'])
x_train, x_test, y_train, y_test, indx_train, indx_test = train_test_split(x, y, indices, test_size=0.20, random_state=rand_st)

#instantiate
linreg = LinearRegression()

#model-training/fitting
model = linreg.fit(x_train,y_train)

#print intercept & coefficients
print(linreg.intercept_) #the trailing _ means that this attribute was ESTIMATED
print(linreg.coef_)

y_pred = linreg.predict(x_test)

"""Therefore, our function is as follows:
$$\widehat{NDVI} = 0.5758681819753635 + (-0.00149881) \times r + (0.00184785) \times g + (-0.00094755) \times b$$
"""

### CREATE SUMMARY DF WITH RGB & PRED
residuals = y_test - y_pred
compare_df = pd.DataFrame({'ID':indx_test,'Actual': y_test, 'Predicted': y_pred, 'Residuals':residuals, 'Absolute Val of Residuals':abs(residuals)})

rgb_and_pred_ndvi = compare_df[['ID','Actual','Predicted']]

mini_rgb_df = []
def get_rgb_vals_from_df(ID):
  Med_r = rgb_vals['Median r'][ID]
  Med_g = rgb_vals['Median g'][ID]
  Med_b = rgb_vals['Median b'][ID]
  dict_row = {'Med r':Med_r, 'Med g':Med_g, 'Med b':Med_b}
  return(dict_row)

for i in rgb_and_pred_ndvi['ID']:
  mini_rgb_df.append(get_rgb_vals_from_df(i))

mini_rgb_df = pd.DataFrame(mini_rgb_df)

rgb_and_pred_ndvi['Med r'] = mini_rgb_df['Med r']
rgb_and_pred_ndvi['Med g'] = mini_rgb_df['Med g']
rgb_and_pred_ndvi['Med b'] = mini_rgb_df['Med b']
rgb_and_pred_ndvi['Predicted NDVI'] = rgb_and_pred_ndvi['Predicted']

rgb_and_pred_ndvi = rgb_and_pred_ndvi[['ID','Med r', 'Med g', 'Med b', 'Predicted NDVI']]
rgb_and_pred_ndvi.columns = ['ID','Med r', 'Med g', 'Med b', 'Predicted NDVI']


'''
################################################################################
#        Section 8 -- Evaluating accuracy of our Multiple Regression Model     #
################################################################################
'''

## **Step 1** -- Visualizing True vs Predicted NDVI
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df1.head(25)

plt.scatter(y_test, y_pred,c='grey')
plt.title("Scatterplot Comparing Actual and Predicted NDVI")
plt.xlabel("Actual NDVI")
plt.ylabel("Predicted NDVI")
plt.savefig("Scatterplot Comparing Actual and Predicted NDVI.png")
files.download("Scatterplot Comparing Actual and Predicted NDVI.png")
plt.show()

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.title("Comparing Actual and Predicted NDVI Values")
plt.savefig("Comparing Actual and Predicted NDVI Values.png")
files.download("Comparing Actual and Predicted NDVI Values.png")
plt.show()

fig = plt.figure(figsize=(10,6))
plt.scatter(df1.index, df1['Actual'], color='blue', marker='o', label='Actual')
plt.scatter(df1.index, df1['Predicted'], color='orange', marker='*', label='Predicted')
plt.xticks(np.arange(0, 25, 1.0))
plt.ylim(0)
plt.legend(loc="upper right")
plt.title("Comparing Actual and Predicted NDVI Values -- 1st 25 Test Values")
plt.ylabel("NDVI")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.savefig("Comparing Actual and Predicted NDVI Values -- 1st 25 Test Values.png")
files.download("Comparing Actual and Predicted NDVI Values -- 1st 25 Test Values.png")
plt.show()

fig = plt.figure(figsize=(10,6))
plt.scatter(compare_df.index, compare_df['Actual'], color='blue', marker='o', label='Actual')
plt.scatter(compare_df.index, compare_df['Predicted'], color='orange', marker='*', label='Predicted')
plt.xticks(np.arange(0, 500, 50.0))
plt.ylim(0)
plt.legend(loc="upper right")
plt.title("Comparing Actual and Predicted NDVI Values -- All Test Values")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.ylabel("NDVI")
plt.savefig("Comparing Actual and Predicted NDVI Values -- All Test Values.png")
files.download("Comparing Actual and Predicted NDVI Values -- All Test Values.png")
plt.show()

residuals = y_test - y_pred
fig = plt.figure(figsize=(10,6))
plt.title("Residual Histogram")
plt.hist(residuals,color='grey')
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.savefig("Residuals Histogram.png")
files.download("Residuals Histogram.png")
plt.show()

residuals = y_test - y_pred
fig = plt.figure(figsize=(10,6))
plt.title("Residual ScatterPlot")
plt.scatter(y_pred, residuals,color='grey',marker='1' )
plt.xlabel("Predicted NDVI")
plt.ylabel("Residual")
plt.axhline(y=0, color='k', ls='--')
plt.savefig("Residuals Scatterplot.png")
files.download("Residuals Scatterplot.png")
plt.show()

## **Step 2** -- A Closer Look at Residuals
# > Create residuals_df with following columns: index, y_test, y_pred, residual, absolute value of residual
df1['Absolute Val of Residuals'] = compare_df['Absolute Val of Residuals']

##**Step 3** -- Extracting examples of great, ok, and bad predictions
def get_img_from_resid(target_resid):
  pic_loc = compare_df[ (compare_df['Absolute Val of Residuals'] == target_resid ) ].index
  pic_loc = pic_loc[0]

  print("Actual NDVI:",compare_df['Actual'][pic_loc])
  print("Predicted NDVI:", compare_df['Predicted'][pic_loc])
  print("Absolute Value of Residual: ", target_resid)

  id_pic = int(compare_df['ID'][pic_loc])
  latitude = globe_df['Latitude'][id_pic]
  longitude = globe_df['Longitude'][id_pic]
  print('Latitude: "{la}", Longitude: "{lo}"'.format(la = latitude,lo = longitude))
  globe_pic_loc = globe_df['Measured Value'][id_pic]
  img = get_img_from_url(globe_pic_loc)
  convert_to_thumbnail(img,(500,500,3))
  return img

# Finding pics for 1st 25 test images
org_compare_df = compare_df.sort_values(by=['Absolute Val of Residuals'])
abs_resid = np.array(org_compare_df['Absolute Val of Residuals'])

# **Sort df1 in ascending order for Absolute Value of Residual**
df1 = df1.sort_values(by=['Absolute Val of Residuals'])
df1.reset_index(inplace=True, drop=True)

###**Step 2.A** -- Example of a Great Prediction"""
min = np.min(abs_resid)
image_best = get_img_from_resid(min)

###**Step 2.B** -- Example of a Mediocre Prediction"""
last_index = org_compare_df.shape[0] - 1
med_index = int((last_index - 0)/2 + 1)
med_resid = org_compare_df['Absolute Val of Residuals'][med_index]
image_ok = get_img_from_resid(med_resid)

###**Step 2.C** -- Example of a Not-so-great Prediction"""
max = abs_resid[len(abs_resid)-4]
image_worst = get_img_from_resid(max)

###**General Testing**
abs_resid = np.array(compare_df['Absolute Val of Residuals'])
target_resid = np.max(abs_resid)
pic_loc = compare_df[ (compare_df['Absolute Val of Residuals'] == target_resid ) ].index
pic_loc = pic_loc[0]

print("Actual:",compare_df['Actual'][pic_loc])
print("Predicted:", compare_df['Predicted'][pic_loc])
print("Absolute Value of Residual: ", target_resid)

id_pic = int(compare_df['ID'][pic_loc])
globe_pic_loc = globe_df['Measured Value'][id_pic]
img = get_img_from_url(globe_pic_loc)

"""##**Step 4** - Standard Evaluation Metrics Calculations

###Linear Regression Evaluation Metrics
1. ***MEAN ABSOLUTE ERROR (MAE)*** -- mean of absolute value of errors
$$MAE = \frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
2. ***MEAN SQUARED ERROR (MSE)*** -- mean of the squared errors
$$MSE = \frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
3. ***ROOT MEAN SQUARED ERROR (RMSE)*** --  square root of MSE
$$RMSE = \sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
"""
#Standard Deviation
np.std(y_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""#Potential Extensions
> Create a GUI that immediately predicts NDVI from image

> Examine other curve-fitting models and compare results

> Explore relationships between feature classifciation and r, g, and b ratios

"""
