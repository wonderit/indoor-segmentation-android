/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.env.SplitTimer;

import java.util.ArrayList;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class TensorFlowDeeplabV3PlusDetector implements SegmentationClassifier {
  private static final Logger LOGGER = new Logger();

  protected static final boolean SAVE_PREVIEW_BITMAP = false;

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 5;

  private static final int NUM_CLASSES = 20;

  private static final int NUM_BOXES_PER_BLOCK = 5;

  // TODO(andrewharp): allow loading anchors and classes
  // from files.
  private static final double[] ANCHORS = {
    1.08, 1.19,
    3.42, 4.41,
    6.63, 11.38,
    9.42, 5.11,
    16.62, 10.52
  };

  private static final int[][] LABEL_COLORS = {
          {0, 0, 0},
            {180, 120, 120},
            {  6, 230, 230},
            { 80,  50,  50},
            {  4, 200,   3},
            {120, 120,  80},
            {140, 140, 140},
            {204,   5, 255},
            {230, 230, 230},
            {  4, 250,   7},
            {224,   5, 255},
            {235, 255,   7},
            {150,   5,  61},
            {120, 120,  70},
            {  8, 255,  51},
            {255,   6,  82},
            {143, 255, 140},
            {204, 255,   4},
            {255,  51,   7},
            {204,  70,   3},
            {  0, 102, 200},
            { 61, 230, 250},
            {255,   6,  51},
            { 11, 102, 255},
            {255,   7,  71},
            {255,   9, 224},
            {  9,   7, 230},
            {220, 220, 220},
            {255,   9,  92},
            {112,   9, 255},
            {  8, 255, 214},
            {  7, 255, 224},
            {255, 184,   6},
            { 10, 255,  71},
            {255,  41,  10},
            {  7, 255, 255},
            {224, 255,   8},
            {102,   8, 255},
            {255,  61,   6},
            {255, 194,   7},
            {255, 122,   8},
            {  0, 255,  20},
            {255,   8,  41},
            {255,   5, 153},
            {  6,  51, 255},
            {235,  12, 255},
            {160, 150,  20},
            {  0, 163, 255},
            {140, 140, 140},
            {250,  10,  15},
            { 20, 255,   0},
            { 31, 255,   0},
            {255,  31,   0},
            {255, 224,   0},
            {153, 255,   0},
            {  0,   0, 255},
            {255,  71,   0},
            {  0, 235, 255},
            {  0, 173, 255},
            { 31,   0, 255},
            { 11, 200, 200},
            {255,  82,   0},
            {  0, 255, 245},
            {  0,  61, 255},
            {  0, 255, 112},
            {  0, 255, 133},
            {255,   0,   0},
            {255, 163,   0},
            {255, 102,   0},
            {194, 255,   0},
            {  0, 143, 255},
            { 51, 255,   0},
            {  0,  82, 255},
            {  0, 255,  41},
            {  0, 255, 173},
            { 10,   0, 255},
            {173, 255,   0},
            {  0, 255, 153},
            {255,  92,   0},
            {255,   0, 255},
            {255,   0, 245},
            {255,   0, 102},
            {255, 173,   0},
            {255,   0,  20},
            {255, 184, 184},
            {  0,  31, 255},
            {  0, 255,  61},
            {  0,  71, 255},
            {255,   0, 204},
            {  0, 255, 194},
            {  0, 255,  82},
            {  0,  10, 255},
            {  0, 112, 255},
            { 51,   0, 255},
            {  0, 194, 255},
            {  0, 122, 255},
            {  0, 255, 163},
            {255, 153,   0},
            {  0, 255,  10},
            {255, 112,   0},
            {143, 255,   0},
            { 82,   0, 255},
            {163, 255,   0},
            {255, 235,   0},
            {  8, 184, 170},
            {133,   0, 255},
            {  0, 255,  92},
            {184,   0, 255},
            {255,   0,  31},
            {  0, 184, 255},
            {  0, 214, 255},
            {255,   0, 112},
            { 92, 255,   0},
            {  0, 224, 255},
            {112, 224, 255},
            { 70, 184, 160},
            {163,   0, 255},
            {153,   0, 255},
            { 71, 255,   0},
            {255,   0, 163},
            {255, 204,   0},
            {255,   0, 143},
            {  0, 255, 235},
            {133, 255,   0},
            {255,   0, 235},
            {245,   0, 255},
            {255,   0, 122},
            {255, 245,   0},
            { 10, 190, 212},
            {214, 255,   0},
            {  0, 204, 255},
            { 20,   0, 255},
            {255, 255,   0},
            {  0, 153, 255},
            {  0,  41, 255},
            {  0, 255, 204},
            { 41,   0, 255},
            { 41, 255,   0},
            {173,   0, 255},
            {  0, 245, 255},
            { 71,   0, 255},
            {122,   0, 255},
            {  0, 255, 184},
            {  0,  92, 255},
            {184, 255,   0},
            {  0, 133, 255},
            {255, 214,   0},
            { 25, 194, 194},
            {102, 255,   0},
            { 92,   0, 255}
  };

    private static final int[][] LABEL_COLORS_OPTIMIZED = {
            {0, 0, 0},
            {120, 120, 120},  //wall
            { 70,  70,  70},  // Others
            {  6, 230, 230},  //sky
            { 80,  50,  50},  //flooring
            {  4, 200,   3},  // tree
            {120, 120,  80},  // ceiling
            { 80,  50,  50},  // route = floor
            {204,   5, 255},  // furniture (bed)
            {120, 120, 120},  // window = wall
            {  4, 250,   7},  // grass
            {204,   5, 255},  // furniture (cabinet)
            {235, 255,   7},  // pavement
            {150,   5,  61},  // soul
            { 80,  50,  50},  // ground = floor
            {  8, 255,  51},  // door
            {204,   5, 255},  // furniture (table)
            {143, 255, 140},  // mount
            {  4, 200,   3},  // life plant = tree
            {204,   5, 255},  // furniture (pall)
            {204,   5, 255},  // furniture (chair)
            { 70,  70,  70},  // Others
            { 61, 230, 250},  // water
            {255,   6,  51},  // picture
            { 11, 102, 255},  // lounge
            {204,   5, 255},  // furniture (shelf)
            {255,   9, 224},  // house
            { 70,  70,  70},  // Others
            {220, 220, 220},  // mirror
            {255,   9,  92},  // carpeting
            { 80,  50,  50},  // field = floor
            {204,   5, 255},  // furniture (armchair)
            {204,   5, 255},  // furniture (seat)
            {120, 120, 120},  //fence = wall
            {204,   5, 255},  // furniture (desk)
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  0, 20,  255},  // box
            {120, 120, 120},  //pillar = wall
            {120, 120, 120},  //sign board = wall
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  0, 20,  255},  // box = icebox
            { 70,  70,  70},  // Others
            { 80,  50,  50},  // path = floor
            { 31,  0, 255},  // steps = staircase
            { 80,  50,  50},  // runway = floor
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 31,  0, 255},  // staircase
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  4, 200,   3},  // flower = tree
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  4, 200,   3},  // plant = tree
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  0,  20, 255},  // box
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 31,   0, 255},  // stairway = staircase
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 31,   0, 255},  // stair = staircase
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  4, 200,   3},  // flowerpot = tree
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  4, 200,   3},  // vase = tree
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {  0,  20, 255},  // box
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            {120, 120, 120},  //sign board = wall
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70},  // Others
            { 70,  70,  70}  // Others
    };

  private static final String[] LABELS = {
          "other",
          "wall",
          "edifice",
          "sky",
          "flooring",
          "tree",
          "ceiling",
          "route",
          "bed ",
          "window ",
          "grass",
          "cabinet",
          "pavement",
          "soul",
          "ground",
          "door",
          "table",
          "mount",
          "life",
          "pall",
          "chair",
          "motorcar",
          "water",
          "picture",
          "lounge",
          "shelf",
          "house",
          "sea",
          "mirror",
          "carpeting",
          "field",
          "armchair",
          "seat",
          "fencing",
          "desk",
          "stone",
          "press",
          "lamp",
          "tub",
          "rail",
          "cushion",
          "stand",
          "box",
          "pillar",
          "sign",
          "dresser",
          "counter",
          "sand",
          "sink",
          "skyscraper",
          "fireplace",
          "icebox",
          "stand",
          "path",
          "steps",
          "runway",
          "vitrine",
          "table",
          "pillow",
          "screen",
          "staircase",
          "river",
          "span",
          "bookcase",
          "screen",
          "table",
          "throne",
          "flower",
          "book",
          "hill",
          "bench",
          "countertop",
          "stove",
          "tree",
          "island",
          "system",
          "chair",
          "boat",
          "bar",
          "machine",
          "shanty",
          "vehicle",
          "towel",
          "source",
          "motortruck",
          "tower",
          "pendent",
          "sunblind",
          "lamp",
          "kiosk",
          "box",
          "plane",
          "track",
          "clothes",
          "pole",
          "soil",
          "handrail",
          "stairway",
          "hassock",
          "bottle",
          "sideboard",
          "card",
          "stage",
          "van",
          "ship",
          "fountain",
          "transporter",
          "canopy",
          "machine",
          "toy",
          "natatorium",
          "stool",
          "cask",
          "handbasket",
          "falls",
          "shelter",
          "bag",
          "motorbike",
          "cradle",
          "oven",
          "ball",
          "food",
          "stair",
          "tank",
          "marque",
          "oven",
          "flowerpot",
          "fauna",
          "cycle ",
          "lake",
          "machine",
          "screen",
          "cover",
          "sculpture",
          "hood",
          "sconce",
          "vase",
          "stoplight",
          "tray",
          "bin",
          "fan",
          "dock",
          "screen",
          "plate",
          "device",
          "board",
          "shower",
          "radiator",
          "glass",
          "clock",
          "flag"
  };

  // Config values.
  private String inputName;
  private int inputSize;

  // Pre-allocated buffers.
  private int[] intValues;
  private byte[] byteValues;
  private String[] outputNames;
  private int[] colorRGB;
  private boolean logStats = true;

  private TensorFlowInferenceInterface inferenceInterface;

  /** Initializes a native TensorFlow session for classifying images. */
  public static SegmentationClassifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final int inputSize,
      final String inputName,
      final String outputName) {
    TensorFlowDeeplabV3PlusDetector d = new TensorFlowDeeplabV3PlusDetector();
    d.inputName = inputName;
    d.inputSize = inputSize;

    // Pre-allocate buffers.
    d.outputNames = outputName.split(",");
    d.intValues = new int[inputSize * inputSize];
    d.byteValues = new byte[d.inputSize * d.inputSize * 3];
    d.colorRGB = new int[3];
//    d.blockSize = blockSize;

    d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

    final Graph g = d.inferenceInterface.graph();

    d.inputName = "ImageTensor";
    // The inputName node has a shape of [N, H, W, C], where
    // N is the batch size
    // H = W are the height and width
    // C is the number of channels (3 for our purposes - RGB)
    final Operation inputOp = g.operation(d.inputName);
    if (inputOp == null) {
      throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
    }
    d.inputSize = inputSize;
    // The outputScoresName node has a shape of [N, NumLocations], where N
    // is the batch size.
    final Operation outputOp1 = g.operation("SemanticPredictions");
    if (outputOp1 == null) {
      throw new RuntimeException("Failed to find output Node 'SemanticPredictions'");
    }

    return d;
  }

  private TensorFlowDeeplabV3PlusDetector() {}

  private float expit(final float x) {
    return (float) (1. / (1. + Math.exp(-x)));
  }

  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

  public Bitmap recognizeImage(final Bitmap bitmap) {
    final SplitTimer timer = new SplitTimer("recognizeImage");

    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    LOGGER.i("PIXEL LENGTH !!  %d", intValues.length);

    for (int i = 0; i < intValues.length; ++i) {
      byteValues[i * 3 + 2] = (byte) (intValues[i] & 0xFF);
      byteValues[i * 3 + 1] = (byte) ((intValues[i] >> 8) & 0xFF);
      byteValues[i * 3 + 0] = (byte) ((intValues[i] >> 16) & 0xFF);
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    inferenceInterface.feed(inputName, byteValues, 1, inputSize, inputSize, 3);
    Trace.endSection();

    timer.endSplit("ready for inference");

    // Run the inference call.
    Trace.beginSection("run");

    LOGGER.i("output names %s ", outputNames);
    inferenceInterface.run(outputNames);
    Trace.endSection();

    timer.endSplit("ran inference");

    // Copy the output Tensor back into the output array.
    Trace.beginSection("fetch");
    final long[] output = new long[inputSize * inputSize];
    final int[] outputIntArray = new int[inputSize * inputSize];

    inferenceInterface.fetch(outputNames[0], output);
    Trace.endSection();

    Bitmap segBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);

    for (int i = 0; i < output.length; i++) {
        colorRGB = LABEL_COLORS_OPTIMIZED[(int) output[i]];
        outputIntArray[i] = Color.argb(200, colorRGB[0],colorRGB[1],colorRGB[2]);
    }
    segBitmap.setPixels(outputIntArray, 0, inputSize, 0, 0, inputSize, inputSize);
    // For examining the actual TF output.
//    if (SAVE_PREVIEW_BITMAP) {
//      ImageUtils.saveBitmap(segBitmap);
//    }

    LOGGER.i("output size : %d", output.length);
    timer.endSplit("decoded results");

    Trace.endSection(); // "recognizeImage"

    timer.endSplit("processed results");

    return segBitmap;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {
    this.logStats = logStats;
  }

  @Override
  public String getStatString() {
    return inferenceInterface.getStatString();
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }


}
