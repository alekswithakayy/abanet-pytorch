# ABANet - Animal Behaviour Analysis Network
Tracking and classifying animals in videos using weakly supervised learning

## Introduction

Camera traps have recently become a popular method of capturing animal behaviour in the wild. While an effective means of gathering data on various species, analysis of the thousands of resulting photos and videos has become increasingly difficult for ecologists. ABANet is an attempt to ease the burden of purely manual analysis by automating parts of the process through deep learning. 

## Objectives

ABANet is currently being developed to measure the fear response of various African species' after they are exposed to sound stimuli. Once a camera trap detects movement and begins filming, a playback of birds (control), lions, humans or gunshots is started. 

In order to automatically measure the animals' response, ABANet aims to find, track and classify animal instances within a video. Since movement is the primary proxy human analysts use to determine behaviour, this method should provide a good intial approximation of the animals' behavioural  response. This function would also allow us to filter out undesirable images and videos (such as empty frames) to reduce the amount data requiring human analysis. Longer-term goals include classifying animal posture (head up, head down, etc.) to act as supplemental behavioural clues. 

## Current Results

Currently, ABANet is capable of performing classification and weak localization of animals within video frames. Our dataset consists only of image level labels, hence our models are trained exclusively through weak supervision.

![](/assets/hyena.gif)

## Future Work

In order to perform weakly-supervised instance segmentation and classification, we are currently implementing the [Inter-Pixel Relation Network](https://arxiv.org/abs/1904.05044).
