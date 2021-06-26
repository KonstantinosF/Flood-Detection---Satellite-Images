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


<b><i>Literature:</i></b>
<ol>
  <li>Flood Detection in Time Series of Optical and SAR Images, C. Rambour,N. Audebert,E. Koeniguer,B. Le Saux,  and M. Datcu, ISPRS - International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2020, 1343--1346</li>

  <li>The Multimedia Satellite Task at MediaEval2019, Bischke, B., Helber, P., Schulze, C., Srinivasan, V., Dengel, A.,Borth, D., 2019, In Proc. of the MediaEval 2019 Workshop</li>
</ol>

<b><i>Links:</b></i>

[1] [http://registry.mlhub.earth/10.21227/w6xz-s898/?fbclid=IwAR2SLWlo24EK2rvBVlpJlXXn9A9jhvszTeID9iHXDbZxbxVM1Ak8uSGbkBM]

[2] [https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection]
