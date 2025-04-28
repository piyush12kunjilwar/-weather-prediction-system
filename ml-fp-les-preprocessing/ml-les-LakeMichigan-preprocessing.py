#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: right">INFO 6106 Machine Learning Final Project</div>
# <div style="text-align: right">Dino Konstantopoulos 5 April 2024</div>

# # Traverse City
# Traverse City is a Lake Michigan coastal city that get a lot of Lake-Effect snow.
# 
# We attempt to verify that cloud sequences are contiguous

# In[1]:


import os
import pandas as pd
import numpy as np
import pickle
import ast

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# I think this is just winter months:

# In[6]:


les = pd.read_csv(r'D:\user\docs\NU\_info6106\2006Fall_2017Spring_GOES_meteo_combined.csv')
les.head()


# # EDA

# In[7]:


column_names = les.columns.tolist()
print(column_names)


# In[7]:


# Renaming
les.rename(columns={ "Temp (F)": "Temp_F", "RH (%)": "RH_pct",
                   "Dewpt (F)" : "Dewpt_F", "Wind Spd (mph)" : "Wind_Spd_mph",
                   "Wind Direction (deg)" : "Wind_Direction_deg", "Peak Wind Gust(mph)" : "Peak_Wind_Gust_mph",
                   "Low Cloud Ht (ft)" : "Low_Cloud_Ht_ft", "Med Cloud Ht (ft)" : "Med_Cloud_Ht_ft",
                   "High Cloud Ht (ft)" : "High_Cloud_Ht_ft", "Visibility (mi)" : "Visibility_mi",
                   "Atm Press (hPa)" : "Atm_Press_hPa", "Sea Lev Press (hPa)" : "Sea_Lev_Press_hPa",
                   "Altimeter (hPa)" : "Altimeter_hPa", "Precip (in)" : "Precip_in",
                   "Wind Chill (F)" : "Wind_Chill_F", "Heat Index (F)" : "Heat_Index_F",
                   } , inplace = True)

les.head()


# #### Missing value handling in dataframe
# - As per the abbr in the table:
#     - `m` or `M`: Data is missing
#     - `NC`: Wind Chill/Heat Index do not meet the required thresholds to be calculated
# 
# Replace the missing values with 0.

# In[8]:


# Replace with 0
les = les.replace(['m', 'M'], '0')


# #### Drop **Wind_Chill_F** and **Heat_Index_F** due to a large number of NC values

# In[9]:


les = les.drop(['Wind_Chill_F', 'Heat_Index_F'], axis=1)
les = les.reset_index(drop=True)


# In[10]:


def missing_values(df):
    total_null = df.isna().sum()
    percent_null = total_null / df.count() # Total count of null values / Total count of values
    missing_data = pd.concat([total_null, percent_null], axis = 1, keys = ['Total Null', 'Percentage Null'])
    return missing_data

missing_values(les)


# Dropping null values:

# In[11]:


les = les.dropna()
print('Total observation count after missing value treatment: ', len(les))


# >**Note to self**: Next run, replace NA with 0 because we may actually have erased too many records by dropping NAs...

# #### Changing Datatype

# In[12]:


les.info()


# In[13]:


# Using apply method
columns = les.columns
les[columns[8:]] = les[columns[8:]].apply(pd.to_numeric, errors='coerce')


# In[14]:


les.info()


# #### Dropping data for the night-time
# We focus on data from **14:00 UTC to 21:00 UTC**, when there is enough sunlight to generate reflections and capture useful information. This time window provides valid data for the experiment and can be used to extract important insights from Lake Michigan and its surrounding areas.
# 
# `14:00 UTC is 10:00am EST and 21:00 UTC is 5:00pm EST.`

# In[15]:


filtered_les = les.loc[(les['Time_UTC'] >= '14:00')
                     & (les['Time_UTC'] <= '21:00')]
filtered_les


# In[16]:


filtered_les = filtered_les.reset_index(drop=True)
filtered_les.head()


# In[17]:


# Summary
filtered_les.describe()


# In[18]:


print('Total observations: ', filtered_les.shape[0])
print('Total number of features: ', filtered_les.shape[1])


# In[8]:


data_sample = les['Lake_data_1D'][16]
data_sample


# # Cloud Imagery

# In[19]:


def arrays_2_png(lat, lon, val, fig_name):
    status_code = -1
    # Here it starts
    if len(lat) == len(lon) == len(val):
        plt.figure(figsize=(10, 10))
        plt.scatter(lon, lat, c=val, cmap=cm.gray, marker='s')
        plt.colorbar(orientation='vertical')
        plt.savefig('D:/user/docs/NU/_Noctis/lake-michigan-images/' + fig_name + '.png')
        status_code = 0
    else:
        status_code = 255

    return status_code


# In[20]:


df_lat_lon = df_lat_lon = pd.read_csv(
    r'D:\user\docs\NU\_Noctis\data\lat_long_1D_labels_for_plotting.csv')
df_lat_lon.head(5)


# In[21]:


df_lat_lon.shape


# In[22]:


lat_lst = df_lat_lon['latitude'].to_list()
lon_lst = df_lat_lon['longitude'].to_list()


# In[48]:


data_sample = filtered_les['Lake_data_1D'][16]
data_sample


# In[49]:


data_sample2 = filtered_les['Lake_data_2D'][16]
data_sample2


# # 1D data conversion 

# In[50]:


import ast

data_sample_lst = ast.literal_eval(data_sample)
data_sample_lst[0:10]


# In[51]:


ldata_sample = data_sample.strip('][').split(', ')
ldata_sample[0:10]


# In[52]:


data_sample_lst2 = [float(el) for el in ldata_sample]
data_sample_lst2[0:10]


# In[53]:


data_sample_lst2 = [float(el) for el in filtered_les['Lake_data_1D'][16].strip('][').split(', ')]
data_sample_lst2[0:10]


# In[54]:


data_sample_lst == data_sample_lst2


# # Plotting 1D data

# In[34]:


arrays_2_png(lat_lst, lon_lst, data_sample_lst, 'sample')


# # goes11.2008.11.03.1600

# In[56]:


filtered_les.loc[5177]


# In[88]:


arrays_2_png(lat_lst, lon_lst, ast.literal_eval(filtered_les['Lake_data_1D'][5177]), 'sample')


# In[58]:


les['Lake_data_1D'][5177].strip('][').split(', ')


# In[60]:


def rectify(crap_string):
    return [0.0 if el == 'nan' else float(el) for el in crap_string.strip('][').split(', ')]


# In[87]:


arrays_2_png(lat_lst, lon_lst, 
             [0.0 if el == 'nan' else float(el) for el in filtered_les['Lake_data_1D'][5177].strip('][').split(', ')], 
             'sample')


# In[63]:


arrays_2_png(lat_lst, lon_lst, 
             rectify(filtered_les['Lake_data_1D'][5177]), 
             'sample')


# In[66]:


from IPython.display import Image
Image("D:/user/docs/NU/_Noctis/original-images/goes11.2008.01.10.1600.v01.nc.png")


# Ok, this looks good.

# # Image generation
# We will generate 64 $\times$ 64 images for each daytime Cloud frame.
# 
# The images are pretty large and take up a lot of memory and processing time for the network, so we resize them into 64 x 64 pixels. Then, we convert the images into grayscale and save them for training. 
# 
# The function below removes the colormap and axis, so that clean images can be stored to train the models:

# In[76]:


# Remove the colormap and axis to clean images
def arrays_2_png_data(lat, lon, val, fig_name):
    status_code = -1

    if len(lat) == len(lon) == len(val):
        plt.figure(figsize=(10, 10))
        plt.scatter(lon, lat, c=val, cmap=cm.gray, marker='s')
        plt.axis('off')
        plt.savefig(f'D:/user/docs/NU/_Noctis/lake-michigan-images/' + fig_name +'.png')
        plt.close()
        status_code = 0
    else:
        status_code = 255

    return status_code


# A small test first:

# In[77]:


for i, row in les.iterrows():
    if i == 10:
        arr = [0.0 if el == 'nan' else float(el) for el in row.Lake_data_1D.strip('][').split(', ')]
        print(arr)
        arrays_2_png_data(lat_lst, lon_lst, arr, str(i))
        break


# OK, this works. Let's read in the 1D column and serialize lake Michigan clouds:

# In[78]:


from tqdm import tqdm
for i, row in tqdm(les.iterrows()):
    if i == 100:
        break


# In[79]:


from tqdm import tqdm
for i, row in tqdm(filtered_les.iterrows()):

    try:
        #arr = np.array(eval(row.Lake_data_1D))
        arr = [0.0 if el == 'nan' else float(el) for el in row.Lake_data_1D.strip('][').split(', ')]
        arrays_2_png_data(lat_lst, lon_lst, arr, str(i))
    except: # If no data is available (fill with zeros)
        #txt = row.Lake_data_1D
        #txt = txt.replace('nan', '0')
        #arr = np.array(eval(txt))
        print("oopsie at row:", i)


# Interesting... Looking at the folder, images around image #12921 are very incomplete. Memory issue? Let
# stry regenerating that image:

# In[80]:


for i, row in tqdm(les.iterrows()):
    try:
        if 12921 == i:
            #arr = np.array(eval(row.Lake_data_1D))
            arr = [0.0 if el == 'nan' else float(el) for el in row.Lake_data_1D.strip('][').split(', ')]
            arrays_2_png_data(lat_lst, lon_lst, arr, 'sample')
    except: # If no data is available (fill with zeros)
        #txt = row.Lake_data_1D
        #txt = txt.replace('nan', '0')
        #arr = np.array(eval(txt))
        print("oopsie at row", str(i))


# Yes, that worked! So, it is *likely* a this notebook's memory issue!
# 
# Looking at the containing folder, it looks like images from image #10127 to image #13046 are corrupt!
# 
# Let's regenerate these in a separate folder:

# In[81]:


def arrays_2_png_data_regen(lat, lon, val, fig_name, folder_name):
    status_code = -1

    if len(lat) == len(lon) == len(val):
        plt.figure(figsize=(10, 10))
        plt.scatter(lon, lat, c=val, cmap=cm.gray, marker='s')
        plt.axis('off')
        plt.savefig(folder_name + '/' + fig_name +'.png')
        plt.close()
        status_code = 0
    else:
        status_code = 255

    return status_code


# In[82]:


for i, row in tqdm(les.iterrows()):
    try:
        if 10127 <= i <= 13046:
            #arr = np.array(eval(row.Lake_data_1D))
            arr = [0.0 if el == 'nan' else float(el) for el in row.Lake_data_1D.strip('][').split(', ')]
            arrays_2_png_data_regen(lat_lst, lon_lst, arr, str(i), "D:/user/docs/NU/_Noctis/lake-michigan-images-regen")
    except: # If no data is available (fill with zeros)
        #txt = row.Lake_data_1D
        #txt = txt.replace('nan', '0')
        #arr = np.array(eval(txt))
        print("oopsie at row", str(i))


# Examining the regeneration folder, most of the images look suspiciously like brown noise. For example, image #12969. Let's regenerate that one to verify:

# In[83]:


for i, row in tqdm(les.iterrows()):
    try:
        if i == 12969:
            #arr = np.array(eval(row.Lake_data_1D))
            arr = [0.0 if el == 'nan' else float(el) for el in row.Lake_data_1D.strip('][').split(', ')]
            arrays_2_png_data_regen(lat_lst, lon_lst, arr, 'sample', "D:/user/docs/NU/_Noctis/lake-michigan-images-regen")
    except: # If no data is available (fill with zeros)
        #txt = row.Lake_data_1D
        #txt = txt.replace('nan', '0')
        #arr = np.array(eval(txt))
        print("oopsie at row", str(i))


# Yup, `sample` looks exactly lime image #12969 in the regenerating folder. 
# 
# Note the missing filename and missing data:

# In[84]:


filtered_les.loc[12969]


# In[85]:


filtered_les['Lake_data_1D'][12969]


# So, appears to be a missing data issue?
# 
# When `File_name_for_1D_lake == None`, that means there is no image data, but we keep the meteo data.
# 
# So let's use this band of missing data as the separation between the training set and the validation set!
# 
# Note to myself: IN order to always ensure that data is not corrupt:
# 
# - For *each meteo city*, produce a combined csv just like Traverse City.
# 
# - Then, run logic that goes over *every row* and verifies that the image filename is not null *and* that the 1D data is not made out of a majority of nans.
# 
# - Then, randomly select 100 rows over the entire dataset and produce a 100-row 2-column image collection that plots lake Michigan cloud cover on the right and the original satellite image on the right.
# 
# We need to be able to scan all 100 images and verify that the cloud covers match.

# I copy contents of folder `D:\user\docs\NU\_Noctis\lake-michigan-images-regen` into folder `D:\user\docs\NU\_Noctis\lake-michigan-images`.

# # Removing the 255-level padding around Lake Michigan
# We need to do this *before* we resize the images to 64 $\times$ 64, otherwise we will get artificial aliasing around the lake MIchigan coastline, which will look like spurious Cloud intensity around the coastline!
# 
# We know that image #39 is corrupt: all black. It should give us the shape of Lake Michigan!

# In[192]:


from PIL import Image, ImageOps

img = Image.open('D:/user/docs/NU/_Noctis/lake-michigan-images/39.png')
img


# Let's create a mask that is all ones *over* lake Michigan, and all zeros over land:

# In[193]:


#full = np.full(img.size, 255)
img = ImageOps.grayscale(img)
#mask = (full - img).astype(np.uint8)
#mask = (0 < mask).astype(int)
img = np.asarray(img)
mask = (255 != img).astype(int)
np.nonzero(mask)


# In[194]:


(mask * 255)[150, 553], (mask * 255)[860, 408]


# In[211]:


plt.imshow(mask * 255, interpolation='none')
plt.show()


# Now let's see what Image #6 should really look like, without the spurious full-intensity over land:

# In[207]:


img = Image.open('D:/user/docs/NU/_Noctis/lake-michigan-images/6.png')
img = ImageOps.grayscale(img)
img


# In[212]:


newimg = np.asarray(img) * mask # mask with the lake michigan mask to zero out outside region
plt.imshow(newimg, interpolation='none')
plt.show()


# In[199]:


(newimg)[150, 553], (newimg)[860, 408]


# In[219]:


ImageOps.grayscale(Image.fromarray(newimg))


# And this is how we save the image above:

# In[220]:


ImageOps.grayscale(Image.fromarray(newimg)).save('D:/user/docs/NU/_Noctis/lake-michigan-images/sample.png')


# To combine all ops:

# In[222]:


f_img = 'D:/user/docs/NU/_Noctis/lake-michigan-images/6.png'
g_img = 'D:/user/docs/NU/_Noctis/lake-michigan-images/sample.png'
img = Image.open(f_img)
img = ImageOps.grayscale(img)
newimg = np.asarray(img) * mask # mask with the lake michigan mask to zero out land region
newimg64 = ImageOps.grayscale(Image.fromarray(newimg)).resize((64,64))
newimg64.save(g_img)


# So now let's repeat these operations *prior* to compressing to 64 $\times$ 64:

# # Shrinking to 64 $\times$ 64
# We now resize images to 64 $\times$ 64 in order to reduce network training memory requirements, with zero intensities on land and avoiding aliasing around the lake border:

# In[224]:


from PIL import Image, ImageOps
f = 'D:/user/docs/NU/_Noctis/lake-michigan-images'
g = 'D:/user/docs/NU/_Noctis/lake-michigan-images-64'
for file in tqdm(os.listdir(f)):
    f_img = f + "/" + file
    g_img = g + "/" + file
    img = Image.open(f_img)
    img = ImageOps.grayscale(img)
    newimg = np.asarray(img) * mask # mask with the lake michigan mask to zero out land region
    newimg64 = ImageOps.grayscale(Image.fromarray(newimg)).resize((64,64), Image.ANTIALIAS)
    newimg64.save(g_img)


# I think there's still aliasing on the coastline compared to the original images, but I think this is about the best we can get.

# # Optional: Limiting
filtered_les = filtered_les[:14000]
# # Correlations
# Plotting the pearson correlation plot to visualise the correlation between various features

# In[95]:


# Correlation 
correlation_matrix = filtered_les.corr(method = 'pearson')
plt.subplots(figsize=(15,12))

# Heatmap
sns.heatmap(correlation_matrix, annot = True, cmap = "YlGnBu")
plt.title("Correlation Matrix", size = 12, weight = 'bold')


# **Observations from the above correlation plots:**
# - Few features are very heavily correated with each other
# - We remove the ones that have shown `positive correlation` greater than 0.6
#     - **Temp_F** is highly correlated with **Dewpt_F**
#     - **Wind_Spd_mph** is highly correlated with **Peak_Wind_Gust_mph**
# - We also note some strong `negative correlation`, but all of them are greater than -0.6, hence we do not drop those features
# 
# We can drop the above columns since they imply to the same information, and keeping them as features will increase the model size.

# In[96]:


filtered_les = filtered_les.drop(['Dewpt_F', 'Peak_Wind_Gust_mph'], axis=1)
filtered_les = filtered_les.reset_index(drop=True)

# Information about dataset shape
print('Total observations: ', filtered_les.shape[0])
print('Total number of features: ', filtered_les.shape[1])
filtered_les.head()


# In[154]:


sns.pairplot(filtered_les)


# In[155]:


def distPlot(data):
    cols = data.columns[4:]
    for col in cols:
        sns.histplot(data[col], kde=True)
        plt.show()
        
distPlot(filtered_les)


# In[97]:


filtered_les['LES_Snowfall'].value_counts()


# In[98]:


sns.countplot(x = filtered_les['LES_Snowfall'], palette=["#7fcdbb", "#edf8b1"])


# # Feature engineering: Precipitation

# In[99]:


filtered_les["Precip_in"].value_counts()


# In[100]:


sns.histplot(filtered_les["Precip_in"])


# In[101]:


filtered_les["Precip_in"][filtered_les["Precip_in"] > 0]


# In[102]:


sns.histplot(filtered_les["Precip_in"][filtered_les["Precip_in"] > 0])


# Adding a new column for precipitation:

# In[103]:


filtered_les.loc[filtered_les['Precip_in'] > 0, 'LES_Precipitation'] = 1
filtered_les.loc[filtered_les['Precip_in'] <= 0, 'LES_Precipitation'] = 0
filtered_les


# In[104]:


sns.countplot(x = filtered_les['LES_Precipitation'], palette=["#7fcdbb", "#edf8b1"])


# # Predicting Cloud patterns
# This means we are going to live with the nighttime discontinuity in imagery.
# 
# First, load all 64 $\times$ 64 images, with cropping of an 8-pixel border all around the lake:

# In[225]:


from tqdm import tqdm
import cv2

images = []
for idx in tqdm(range(15959)):
    # im shape -> (64, 64)
    im = cv2.imread('D:/user/docs/NU/_Noctis/lake-michigan-images-64/' + str(idx) + '.png')
    # Storing 1 channel, since the images are grayscale, and cropping
    images.append(im[8:-8,8:-8,0]) 
    # images shape -> (35, 64, 64) 


# In[226]:


plt.imshow(images[146]) 


# In[227]:


plt.imshow(images[147])


# In[228]:


from PIL import Image, ImageOps
Image.open('D:/user/docs/NU/_Noctis/lake-michigan-images-64/147.png')


# In[152]:


full = np.full(im.shape, 255)
mask = (full - img).astype(np.uint8)
mask = (0 < mask).astype(int)
mask


# ## Cloud Sequence Visualization
# 
# Our data consists of sequences of frames, each of which
# are used to predict the upcoming frame. Let's take a look
# at some of these sequential frames.
# 
# >**Note**: Do not run the next cell because it shows an example that includes corrupt images (ones with just nans):

# In[230]:


# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# Plot each of the sequential images for one random data example.
data_choice = np.random.choice(range(len(images)), size=1)[0]
for idx, ax in enumerate(axes.flat):
    ax.imshow(images[data_choice + idx], cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Print information and display the figure.
print(f"Displaying next frames starting at image {data_choice}.")
plt.show()


# Run this one instead, whcih displays a valid sequence of images:

# In[231]:


# Construct a figure on which we will visualize the images.
fig, axes = plt.subplots(4, 5, figsize=(10, 8))

# Plot each of the sequential images for one random data example.
data_choice = np.random.choice(range(len(images)), size=1)[0]
for idx, ax in enumerate(axes.flat):
    ax.imshow(images[data_choice + idx], cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")

# Print information and display the figure.
print(f"Displaying next frames starting at image {data_choice}.")
plt.show()


# Since daytime only consists of 7 hours, this image sequence of length 20 obligatorily includes nighttimes. In other words, there is an image above that jumps over nighttime and thus is more discontinuous in cloud cover.
# 
# 20 images is about 3 days (3 $\times$ 7). 
# 
# As an exercise, let's see if based on 6 hours of cloud cover, we can predict the 7th hour.
# 
# We are going to use 6 sequential images as the input, and the next (shifted by 1) 6 images as output.

# In[ ]:




