// Copyright (c) 2019, the Dart project authors. Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:ffi';
import 'dart:io';

/// TensorFlowLite C library.
DynamicLibrary tflitelib = () {
  // Loads TFLite binary to be able to call every method
  // exposed by it on mobile platforms.
  return Platform.isAndroid
      ? DynamicLibrary.open("libtensorflowlite_jni.so")
      : DynamicLibrary.process();
}();
