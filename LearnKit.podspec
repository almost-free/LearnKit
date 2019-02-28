Pod::Spec.new do |s|
  s.name             = 'LearnKit'
  s.version          = '1.6.0'
  s.summary          = '(DEPRECATED) Wrapper around LearnKit.'

  s.description      = <<-DESC
(DEPRECATED) Wrapper around [LearnKit](https://github.com/mattrajca/LearnKit) to easy its integration in Swift CocoaPods.
                       DESC

  s.homepage         = 'https://github.com/almost-free/LearnKit'
  s.license          = { :type => 'MIT', :file => 'LICENSE' }
  s.author           = { 'Jon Willis + ' => 'indisoluble_dev@me.com' }
  s.source           = { :git => 'https://github.com/almost-free/LearnKit.git' }

  s.platform = :ios
  s.ios.deployment_target = '8.0'

  s.prefix_header_file = 'LearnKit/Prefix.pch'

  s.source_files = 'LearnKit/*.{h,m}',
                   'LearnKit/**/*.{h,m}',
                   'LearnKit/fmincg/*.{h,m}',
                   'LearnKit/liblbfgs/include/*.h',
                   'LearnKit/liblbfgs/lib/*.{h,m}'

  s.exclude_files = 'LearnKit/Core/LNKMatrixImages.{h,m}',
                    'LearnKit/Neural Network/LNKNeuralNetClassifier+Debugging.m',
                    'LearnKit/UI/*.{h,m}'

  #s.requires_arc = 'HRLAlgorithms/Classes/**/*.m'
   s.requires_arc = false

end
