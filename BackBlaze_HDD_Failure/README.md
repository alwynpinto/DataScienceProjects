Capstone Project II Final Report
Back Blaze HDD Failure Prediction
Alwyn Pinto
Date: 26th May 2020

Slides: https://docs.google.com/presentation/d/1ovJAusjnYqhLE3ITKOHhaLnzEabA2MgVMDmFOApV9QI/edit?usp=sharing
Report: https://drive.google.com/file/d/1e59_RJcHyxamELWsbAT4pwoxpoiGn4UR/view?usp=sharing

Abstract:
Using data acquired from the Back Blaze website which includes Date, Serial Number, Model and raw and normalized values of SMART variables recorded since 2013. Using these SMART  raw data and running various models, an attempt is made to predict Hard Drive failures.

Introduction:
Welcome to my project on Prediction of Hard Drive failures. Each day since 2013, a snapshot of the basic drive information is taken along with SMART statistics of each operational drive.
The data reported is as follows:
Date – The date of the file in yyyy-mm-dd format.
Serial Number – The manufacturer-assigned serial number of the drive.
Model – The manufacturer-assigned model number of the drive.
Capacity – The drive capacity in bytes.
Failure – Contains a “0” if the drive is OK. Contains a “1” if this is the last day the drive was operational before failing.
2013-2019 SMART Stats – Rawalues 63 different SMART stats as reported by the given drive. Each value is the number reported by the drive.
A link to the dataset can be found here:
https://www.backblaze.com/b2/hard-drive-test-data.html#overview-of-the-hard-drive-data

Problem Statement:
Given the daily snapshot of each drive in the Back Blaze Hard Drive data center, predict the failure of a hard drive using the SMART statistics recorded by each operational drive.
Back Blaze is currently considering if 3 other SMART metrics can be considered towards the prediction of the failure of hard drives.
Back Blaze 5 SMARTs: smart_5, smart_187, smart_188, smart_197, smart_198
Additional SMARTs: smart_9, smart_12, smart_189

Caveats:
SMART statistics are inconsistent among manufacturers and could mean different metrics for each manufacturer. Back Blaze themselves use only 5 metrics for their analysis.
Ref: https://www.backblaze.com/blog/what-smart-stats-indicate-hard-drive-failures/

Addiional SMART under consideration:
SMART 189 – High Fly Writes. It is a cumulative count of the number of times the recording head “flies” outside its normal operating range.
SMART_12 - Power Cycles. This stat could answer the dilemma between powering off your devices for longevity. Back Blaze does recycle their power every couple of months which is not the same as turning off a machine.
As seen from the correlation, that these 8 metrics have little correlation with each other hence these 5 can be considered for further modelling.
Two datasets are run against various different models to see only these 5 or 8 variables to be considered. One way of approaching this with reason and without causing too much of a negative effect, was by considering only a single manufacturer.

The two dataset under consideration has been:
Back Blaze 5 SMART metrics
Back Blaze 5 SMART + Additional 3 SMART metrics.

