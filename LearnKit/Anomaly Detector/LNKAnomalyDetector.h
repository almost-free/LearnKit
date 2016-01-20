//
//  LNKAnomalyDetector.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

/// A multivariate Gaussian-based anomaly detector.
/// The optimization algorithm for anomaly detectors is ignored and can be `nil`.
/// Predicted values are either 0 or 1, with 1 indicating an anomaly. They are represented with `LNKClass`.
@interface LNKAnomalyDetector : LNKClassifier

/// The default value is 0.01.
@property (nonatomic) LNKFloat threshold;

@end


/* Analysis */

/// Given an unlabeled data matrix and labeled cross-validation matrix, we try to find a sensible anomaly threshold.
/// Note that the two matrices must have the same number of columns.
extern LNKFloat LNKFindAnomalyThreshold(LNKMatrix *matrix, LNKMatrix *cvMatrix);

NS_ASSUME_NONNULL_END
