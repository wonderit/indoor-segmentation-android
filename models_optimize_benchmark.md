# Summarizing Graph
bazel build tensorflow/tools/graph_transforms:summarize_graph && \
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=tensorflow/examples/android/assets/quantized_deeplabv3_mnv2_257.pb

# Optimizing frozen graph
bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=tensorflow/examples/android/assets/frozen_inference_graph_xception_257_10K.pb \
--out_graph=tensorflow/examples/android/assets/optimized_frozen_inference_graph_xception_257_10K.pb --inputs='ImageTensor' --outputs='SemanticPredictions' \
--transforms='
  strip_unused_nodes(type=float, shape="1,257,257,3")
  fold_constants(ignore_errors=true)
  fold_batch_norms
  fold_old_batch_norms'

# Quantizing graph
bazel build tensorflow/tools/graph_transforms:transform_graph && \
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=tensorflow/examples/android/assets/optimized_deeplabv3_mnv2_.pb \
--out_graph=tensorflow/examples/android/assets/quantized_deeplabv3_mnv2.pb \
--inputs='ImageTensor:0' --outputs='SemanticPredictions:0' --transforms='quantize_weights'

# Benchmark Models
bazel-bin/tensorflow/tools/benchmark/benchmark_model \
--graph=tensorflow/examples/android/assets/quantized_deeplabv3_mnv2.pb --show_flops --input_layer=ImageTensor --input_layer_type=uint8 --input_layer_shape=1,513,513,3 --output_layer=SemanticPredictions

# Build tensorflow android library with Bazel
bazel build -c opt --config=armeabi-v7a //tensorflow/examples/android:tensorflow_demo