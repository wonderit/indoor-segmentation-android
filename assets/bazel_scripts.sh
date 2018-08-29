#!/usr/bin/env bash
bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=train/optimized_inception_graph.pb

bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=train/frozen_inference_graph_mnv2_129.pb

bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/frozen_inference_graph_mnv2_129.pb \
--out_graph=train/optimized_frozen_inference_graph_mnv2_129.pb --inputs='ImageTensor' --outputs='SemanticPredictions' \
--transforms='
  strip_unused_nodes(type=float, shape="1,129,129,3")
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms'

  bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/frozen_inference_graph_1M.pb \
--out_graph=train/optimized_frozen_inference_graph_1M.pb --inputs='ImageTensor' --outputs='SemanticPredictions' \
--transforms='
  strip_unused_nodes(type=float, shape="1,129,129,3")
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms'


bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=tensorflow/examples/android/assets/frozen_inference_graph_xception_257_10K.pb \
--out_graph=tensorflow/examples/android/assets/optimized_frozen_inference_graph_mnv2_129.pb --inputs='ImageTensor' --outputs='SemanticPredictions' \
--transforms='
  strip_unused_nodes(type=float, shape="1,129,129,3")
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms'



bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/optimized_frozen_inference_graph_1M.pb \
--out_graph=train/quantized_mnv2_129_1M.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'


bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/optimized_frozen_inference_graph_mnv2_129.pb \
--out_graph=train/quantized_mnv2_129.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=train/quantized_mnv2_129_10K.pb

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=train/quantized_mnv2_129_1M.pb

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=train/quantized_mnv2_129_10K.pb --show_flops --input_layer=ImageTensor --input_layer_type=uint8 --input_layer_shape=1,129,129,3 --output_layer=SemanticPredictions

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=train/quantized_mnv2_129_1M.pb --show_flops --input_layer=ImageTensor --input_layer_type=uint8 --input_layer_shape=1,129,129,3 --output_layer=SemanticPredictions

# mnv2 129 10K end

bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/optimized_inception_graph_mnv2_257_257.pb \
--out_graph=train/quantized_mnv2_257_257_1000.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/optimized_inception_graph_mnv2_257_257.pb \
--out_graph=train/quantized_mnv2_257_257_1000.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=tensorflow/examples/android/assets/optimized_2_mnv2_257_1M.pb \
--out_graph=tensorflow/examples/android/assets/quantized_2_mnv2_257_1M.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=train/optimized_inception_graph_xception_1358.pb \
--out_graph=train/quantized_xception_1358.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=tensorflow/examples/android/assets/optimized_frozen_inference_graph_xception_257_10K.pb

bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=tensorflow/examples/android/assets/frozen_inference_graph.pb --show_flops --input_layer=ImageTensor --input_layer_type=uint8 --input_layer_shape=1,513,513,3 --output_layer=SemanticPredictions

# Benchmark for ssd model
--in_graph=tensorflow/examples/android/assets/quantized_mnv2_1000000.pb
bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=tensorflow/examples/android/assets/frozen_inference_graph_xception_257_10K.pb

bazel build tensorflow/tools/benchmark:benchmark_model && \
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=train/quantized_mnv2_1000000.pb --show_flops --input_layer=ImageTensor --input_layer_type=uint8 --input_layer_shape=1,513,513,3 --output_layer=SemanticPredictions


bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=train/quantized_mnv2_1000000.pb \
  --output_file=train/quantized_mnv2_1000000.tflite \
  --inference_type=FLOAT \
  --input_shape=1,513,513,3 \
  --input_array=ImageTensor \
  --output_array=SemanticPredictions

bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=train/optimized_inception_graph.pb \
  --output_file=train/quantized_xception.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,513,513,3 \
  --input_array=ImageTensor \
  --output_array=SemanticPredictions

bazel clean && \
    bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=armeabi-v7a

# For arm64-v8a
bazel clean && \
    bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
    --crosstool_top=//external:android/crosstool \
    --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
    --cpu=arm64-v8a

bazel build -c opt --config=armeabi-v7a //tensorflow/examples/android:tensorflow_demo

# For arm64-v8a
bazel build -c opt --config=arm64-v8a //tensorflow/examples/android:tensorflow_demo