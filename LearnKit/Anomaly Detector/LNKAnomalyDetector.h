//
//  LNKAnomalyDetector.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// A multivariate Gaussian-based anomaly detector.
/// The optimization algorithm for anomaly detectors is ignored and can be `nil`.
/// Predicted values are either 0 or 1, with 1 indicating an anomaly. They are represented with `LNKClass`.
@interface LNKAnomalyDetector : LNKClassifier

/// The default value is 0.01.
@property (nonatomic) LNKFloat threshold;

@end
