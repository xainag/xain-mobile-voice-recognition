Pod::Spec.new do |s|
  s.name             = 'VendoredTfLite'
  s.version          = '2.1.0'
  s.summary          = 'Vendored TensorFlow Lite binary'
  s.description      = <<-DESC
  TensorFlow Lite is TensorFlow's lightweight solution for Flutter developers. It
  enables low-latency inference of on-device machine learning models with a
  small binary size and fast performance supporting hardware acceleration.
                       DESC
  s.authors          = 'Google Inc.'
  s.license          = { :type => 'Apache' }
  s.homepage         = 'https://github.com/tensorflow/tensorflow'
  s.source           = { :git => 'https://github.com/tensorflow/tensorflow.git', :tag => "v#{s.version}" }
  
  s.dependency 'TensorFlowLiteC', "#{s.version}"
  s.ios.vendored_frameworks = 'TensorFlowLiteC.framework'
  s.ios.deployment_target = '9.0'

  # Flutter.framework does not contain a i386 slice. Only x86_64 simulators are supported.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'VALID_ARCHS[sdk=iphonesimulator*]' => 'x86_64' }
  # Fail early during build instead of not finding the library during runtime
  s.xcconfig = { 'OTHER_LDFLAGS' => '-framework TensorFlowLiteC -all_load' }
end