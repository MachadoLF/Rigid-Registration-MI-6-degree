# Rigid-Registration-MI-6-degree
Algorithm to perform 3D rigid registration considering translation and rotation in a Multi-Stage Approach. 

In this example we will run 3D rigid registration
with Both Mattes, Tsallis, and NormTsallis Metric.
Being Tsallis and NormTsallis two mutual information metrics
implemented similarly to ItkMattesMutualInformationImageMetric Class.
Tsallis and NormTsallis need qT and qR (q-value for Translation and rotation, respectively).
Those parâmeter are used when calculating the similarity between the two images 
By Tsallis and NormTsallis metric.

Use as input images the CT (fixed) and T1 (miving) images in /data.

The code needs the following parameters:
fixedImageFile   movingImageFile   MetricType ('Mattes' 'Tsallis' 'TsallisNorm') 
[q-R] [q-T] [strategy] ('-o' to optimization, '-e' to execution)" 
[save images] ('-s' for saving, Null, for not.)

When using Mattes metric one can pass:
fixedImageFile   movingImageFile  MetricType ('Mattes') [save images] ('-s' for saving, Null, for not.)
