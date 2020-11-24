# Rigid-Registration-MI-6-degree
Algorithm to perform 3D rigid registration considering translation and rotation. 

In this example we will run 3D rigid registration
with Both Mattes, Tsallis, and NormTsallis Metric;

Receives at parameters:
fixedImageFile   movingImageFile   MetricType ('Mattes' 'Tsallis' 'TsallisNorm') 
[qValue]  [strategy] ('-o' to optimization, '-e' to execution)" << std::endl 
[save images] ('-s' for saving, Null, for not.)

One can also pass:
fixedImageFile   movingImageFile  MetricType ('Mattes') [save images] ('-s' for saving, Null, for not.)
