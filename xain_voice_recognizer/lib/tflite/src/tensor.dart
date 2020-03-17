// Copyright (c) 2019, the Dart project authors. Please see the AUTHORS file
// for details. All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:quiver/check.dart';

import 'bindings/tensor.dart';
import 'bindings/types.dart';
import 'ffi/helper.dart';

export 'bindings/types.dart' show TfLiteType;

/// TensorFlowLite tensor.
class Tensor {
  final Pointer<TfLiteTensor> _tensor;

  Tensor(this._tensor) {
    checkNotNull(_tensor);
  }

  /// Name of the tensor element.
  String get name => Utf8.fromUtf8(TfLiteTensorName(_tensor));

  /// Data type of the tensor element.
  TfLiteType get type => TfLiteTensorType(_tensor);

  /// Dimensions of the tensor.
  List<int> get shape => List.generate(
      TfLiteTensorNumDims(_tensor), (i) => TfLiteTensorDim(_tensor, i));

  /// Underlying data buffer as bytes.
  Int8List get data {
    final data = cast<Int8>(TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    return UnmodifiableInt8ListView(
        data.asTypedList(TfLiteTensorByteSize(_tensor)));
  }

  /// Updates the underlying data buffer with new bytes.
  ///
  /// The size must match the size of the tensor.
  set data(Int8List bytes) {
    final tensorByteSize = TfLiteTensorByteSize(_tensor);
    checkArgument(tensorByteSize == bytes.length);
    final data = cast<Int8>(TfLiteTensorData(_tensor));
    checkState(isNotNull(data), message: 'Tensor data is null.');
    final externalTypedData = data.asTypedList(tensorByteSize);
    externalTypedData.setRange(0, tensorByteSize, bytes);
  }

  /// Copies the input bytes to the underlying data buffer.
  // TODO(shanehop): Prevent access if unallocated.
  void copyFrom(Int8List bytes) {
    var size = bytes.length;
    final ptr = allocate<Int8>(count: size);
    final externalTypedData = ptr.asTypedList(size);
    externalTypedData.setRange(0, bytes.length, bytes);
    checkState(TfLiteTensorCopyFromBuffer(_tensor, ptr.cast(), bytes.length) ==
        TfLiteStatus.ok);
    free(ptr);
  }

  /// Returns a copy of the underlying data buffer.
  // TODO(shanehop): Prevent access if unallocated.
  Int8List copyTo() {
    var size = TfLiteTensorByteSize(_tensor);
    final ptr = allocate<Int8>(count: size);
    final externalTypedData = ptr.asTypedList(size);
    checkState(
        TfLiteTensorCopyToBuffer(_tensor, ptr.cast(), 4) == TfLiteStatus.ok);
    // Clone the data, because once `free(ptr)`, `externalTypedData` will be
    // volatile
    final bytes = externalTypedData.sublist(0);
    free(ptr);
    return bytes;
  }

// Unimplemented:
// TfLiteTensorQuantizationParams
}
