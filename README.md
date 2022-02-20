# Disaster Management Toolkit
An application that can perform online disaster assessment


<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]


<!-- ABOUT THE PROJECT -->
## About The Project
-------

The purpose of this hack is to aid in the real-time and rapid disaster identification and alleviation by promptly identifying the degree of damaged buildings and areas using remotely sensed images (satellite images) and also provides online updates of earthquakes and fires on the basis of the current scenario on the field.

Segmentation of remotely sensed images lies at the intersection of the domains of remote sensing and computer vision. It is used to systematically extract information from data collected by various airborne and space-borne sensors, resulting in a simpler representation. This method is used in various applications which include change detection, land cover and land use classification, resource exploration, the study of natural hazards, and mapping. In this work, we will focus on the study of natural hazards, i.e., building a multi-class semantic segmentation model that categories the given post-disaster (earthquakes and forest fires in particular) imagery based on damage. 

Currently, the disaster assessment task is done on satellite images manually by humans or by inspecting the damaged site in person which is both slow, inefficient and dangerous. Our model is able to automate the process and helps the rescue workers provide timely assistance even in hard-to-reach areas.  
During a disaster, rescue authorities would like to prioritise by going to severely affected areas, and our application through deep learning and the web user interface will provide the end-user with enough details and up to date updates to prioritise the areas requiring immediate attention by acquiring images from satellite or drones which are connected to the network. And also, in disasters, people trapped under buildings have very little time to survive and it is imperative to address them as soon as possible.

The model was trained on the Xview2 building damage assessment dataset.

# Network
-------

For EarthQuakes : 
For the given task we propose to go with the traditional U-Net architecture composed with an MRA (Multi-Resolution Analysis ) framework. The U-Net architecture is a simple encoder-decoder fully convolutional pipeline consisting of contracting (encoder) and expanding/extracting (decoder) paths. 
The MRA framework is interspersed into the U-Net Architecture in such a way that it pre-processes the inputs to the network at several stages to increase the contextual overview of the network as the same data on multiple scales is available for feature extraction and learning. 

For Forest-Fire : 
We used a traditional VGG Architecture organized in a U-Net style Encoder Decoder.


## Why MRANet? 
-------

The intuition behind using multi-resolution analysis is that images contain features at different scales important for segmentation, therefore, a multi-resolution analysis (MRA) approach is useful for their extraction since this decomposition allows us to even segment structures of various dimensions and structures with ease.

## Why VGGNet? 
-------

The features for forest fire assessment are not complex so this model was enough for good output.

## Network Architecture
-------
MRA Network : 
<p align="center">
  <img src="assets/network_architecture.png" width="550" height="750" title="network">
</p>

VGG Model :  
<p align="center">
  <img src="assets/vgg_image.png" width="750" height="400" title="network">
</p>

## Loss Function
-------

Categorical cross-entropy was used as the loss function

<p align="center">
  <img src="assets/loss.png" width="350" height="100" title="loss">
</p>

## Dataset
-------

Please Download the Xview2 Earthquake disaster Dataset and save the Images and Labels_json in dataset. Create the groundtruth masks from the json file using Labels_Generate.py and save the labels in folder Labels.


### Setup
-------

```
Will update soon
```


## Run
-------

```
Will update soon
```

## Preview
<p align="center">
  <img src="assets/1.png" width="800" height="350" title="loss">
</p>

<p align="center">
  <img src="assets/2.png" width="800" height="350" title="loss">
</p>

## Model Results

### Earthquake damage assessment

<p align="center">
  <img src="assets/results.png" width="800" height="350" title="loss">
</p>

meanIOU:       88.17% <br>
meanDice:      92.55% <br>
pixelAccuracy: 89.9% <br>
Precision:     90.27% <br>
Recall:        92.98% <br>


### Forest Fire damage assessment

<p align="center">
  <img src="assets/5.png" width="400" height="400" title="VGGMOdel">
</p>


meanIOU:       93.34% <br>
meanDice:      95.27% <br>
pixel accuracy: 93.91% <br>
Precision:     92.17% <br>
Recall:        94.58% <br>


<!-- LICENSE -->
## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->
