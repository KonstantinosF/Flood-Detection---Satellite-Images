# Flood-Detection---Satellite Images
<b><em><i>MSc (Master of Science) in Data Science - NCSR "Demokritos" & UoP</i></em></b> </br>
<b><em><i>Team Members </i></em></b>
  1.  Konstantinos (Kostis) Nikolareas
  2.  Konstantinos (Kostas) Fokeas

This is a Semester Project which aim is to employ a Deep Learning model in order to detect Flood Events from Satellite Images

<em><b><i>General Description</i></b></em>

Dataset's Name: SEN12-FLOOD : A SAR and Multispectral Dataset for Flood Detection </br>
This dataset is composed of co-registered optical and SAR images time series for the detection of flood events.

Time Period: December 2019 up to May 2019

Study Areas: African, Iranian, Australian cities


# Description of the Dataset 
## General 
The observed areas correspond to 337 locations (cities and their surroundings ) in West and SouthEast Africa, Middle-East, and Australia where a flood event occurred during the considered period. The period of acquisition goes from December 2018 to May 2019.
For each location, the following data are provided:

* Time series of Sentinel-2 multispectral images. These images are composed of 12 bands,
at 10m ground-sampling distance and are provided with Level 2A atmospheric correction.
* Time series of Sentinel-1 Synthetic Aperture Radar (SAR) images. The images are provided
with radiometric calibration and range doppler terrain correction based on the SRTM digital
elevation model. For one acquisition, two raster images are available corresponding to the
polarimetry channels VV and VH.
* Time series of binary labels for each image / date: flood or no flood.
## Sentinel 2
![Sentinel 2 Specifications](/images/sentinel_2.png)


![Sentinel 2 Specifications_V2](/images/sentinel_2_specs.png)

## Sentinel 1

### Instruments
Sentinel-1 spacecraft are designed to carry the following instruments:

* A single C-band synthetic-aperture radar (C-SAR) with its electronics. This instrument provides 1 dB radiometric accuracy with a central frequency at 5.405 GHz.[10] The data collected in C-SAR was made to be continuous after the termination of a previous mission (Envisat mission).
* An SDRAM-based Data Storage and Handling Assembly (DSHA), with an active data storage capacity of about 1,443 Gbit (168 GiB), receiving data streams from SAR-SES over two independent links gathering SAR_H and SAR_V polarization, with a variable data rate up to 640 Mbit/s on each link, and providing 520 Mbit/s X-band fixed-user data-downlink capability over two independent channels towards ground.
  

### Characteristics
Specifications of the Sentinel-1 satellites:

* 7 year lifetime (12 years for consumables)
* Launcher: Soyuz
* Near-polar (98.18°) Sun-synchronous orbit
* 693 km (431 mi) altitude
* 12-day repeat cycle
* 175 revolutions per cycle
* 98.6 minute orbital period
* 3-axis altitude stabilization
* 2,300 kg (5,100 lb) launch mass
* 3.9 m × 2.6 m × 2.5 m (12.8 ft × 8.5 ft × 8.2 ft) dimensions

# How the Initial Dataset is Organized
### Sentinel 2
The dataset from Sentinel 2 satellite is splitted into 2 main folders, the one with the actual images and another one with the label.
The image below illustrates how these two folders look like on a local machine

![Images and Labels foldes](/images/Images_Labels.png)

If we then open the folder containing the images the result looks like the following image. Each image is stored in a seperate folder.
The name of the folder indicates the different locations with the prefix "source_1" and the date of sensing.

![Images folder 1](/images/Images_1.png)

The next image demonstrates the contents of one of this folders. As we can see the image is splitted into 13 spectral bands. In the following steps we will have to stack those bands into a single multispectral image.

![Images folder 1](/images/Images_2.png)

# Pre Processing

In order to be able to feed the dataset into our machine learning model we first need to clean it, transforme it and split it into training and testing datasets. As we found out many folders are empty (without spectral bands) while others are containing images whose pixels are zero. Additionaly, we need to stack the spectral bands and create a single image. In our expirement we used the bands 2, 3, 4 and 8 corespoding to blue, green, red and near infrared which share the same spatial resolution. Thus, the first step is to execute the python script named "Pre-Processing"

After having clean the dataset we end up with <b>1947</b> images while the initial number of images-folders was <b>2237</b>.

In order to count the number of files or subfolders use the next bash shell command: </br>
<em><i> $ ls /etc | wc -l </em></i>


# Deap Learning Modelling
## Architecure 
* We used Keras

* the sheme of the model 

Figure

Train dataset size - Validation size
* Train on 1343 samples, validate on 336 samples

Test size
* The number of testing samples is = 268


### Model Summary
![Model Summary](/images/model_summary.png)

### Evaluation


## Feature extraction with VGG ImageNet
VGG Net is the name of a pre-trained convolutional neural network (CNN) invented by Simonyan and Zisserman from Visual Geometry Group (VGG) at University of Oxford in 2014. VGG Net has learned to extract the features (feature extractor) that can distinguish the objects and is used to classify unseen objects. VGG was invented with the purpose of enhancing classification accuracy by increasing the depth of the CNNs. VGG 16 and VGG 19, having 16 and 19 weight layers, respectively, have been used for object recognition. VGG Net takes input of 224×224 RGB images and passes them through a stack of convolutional layers with the fixed filter size of 3×3 and the stride of 1. There are five max pooling filters embedded between convolutional layers in order to down-sample the input representation (image, hidden-layer output matrix, etc.). The stack of convolutional layers are followed by 3 fully connected layers, having 4096, 4096 and 1000 channels, respectively. The last layer is a soft-max layer. Below figure shows VGG network structure.

### VGG Architecture
![VGG Net](/images/VGG_structure.jpeg)

### Evaluation
![Model Summary](/images/VGG_evaluation_1.png)

268/268 [==============================] - 1s 3ms/sample - loss: 5.1529 - acc: 0.6045






## Literature
<ol>
  <li>Flood Detection in Time Series of Optical and SAR Images, C. Rambour,N. Audebert,E. Koeniguer,B. Le Saux,  and M. Datcu, ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2020, 1343--1346</li>

  <li>The Multimedia Satellite Task at MediaEval2019, Bischke, B., Helber, P., Schulze, C., Srinivasan, V., Dengel, A.,Borth, D., 2019, In Proc. of the MediaEval 2019 Workshop</li>
</ol>

<b><i>Links:</b></i>

* [http://registry.mlhub.earth/10.21227/w6xz-s898/?fbclid=IwAR2SLWlo24EK2rvBVlpJlXXn9A9jhvszTeID9iHXDbZxbxVM1Ak8uSGbkBM]

* [https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection]

* [https://keras.io/]

* [https://en.everybodywiki.com/VGG_Net]