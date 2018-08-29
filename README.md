# Tensorflow Android Deeplabv3 with MobileNetV2 as a Backbone.

This app is for indoor semantic image segmentation for Android devices.

## Description

The demo app is based on Tensorflow android sample app with bazel build. 


## Tensorflow Android Sample App. 

Inference is done using the [TensorFlow Android Inference
Interface](../../../tensorflow/contrib/android), which may be built separately
if you want a standalone library to drop into your existing application. Object
tracking and efficient YUV -> RGB conversion are handled by
`libtensorflow_demo.so`.

A device running Android 5.0 (API 21) or higher is required to run the demo due
to the use of the camera2 API, although the native libraries themselves can run
on API >= 14 devices.

## Current samples:

Other samples are deleted. 

SegmentationActivity extends CameraActivity. 

This Activity is based on DetectorActivity of previous TF Sample App.

## Prebuilt Components:

If you just want the fastest path to trying the demo, you may download the
nightly build
[here](https://ci.tensorflow.org/view/Nightly/job/nightly-android/). Expand the
"View" and then the "out" folders under "Last Successful Artifacts" to find
tensorflow_demo.apk and libtensorflow_demo.so file.

libtensorflow_inference.so should be re-compiled due to the tf.slice operation compatibility error of nightly-android build.

libtensorflow_inference.so is already compiled in the library folder. 
 
See [libs/armeabi-v7a](libs/armeabi-v7a) for more details.

Libraries for arm64-v8a device added. [libs/arm64-v8a](libs/arm64-v8a)

## Running the Demo

Once the app is installed it can be started via the "TF Segmentation" which have the orange TensorFlow logo as
their icon.

While running the activities, pressing the volume keys on your device will
toggle debug visualizations on/off, rendering additional info to the screen that
may be useful for development purposes.


## Demo GIF
- crop size : 129 x 129 :
<img src="sample_images/mnv2_129.gif" width="300px">

- crop size : 257 x 257 :
<img src="sample_images/mnv2_257.gif" width="300px">