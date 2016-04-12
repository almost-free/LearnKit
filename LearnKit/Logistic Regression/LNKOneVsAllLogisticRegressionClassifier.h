//
//  LNKOneVsAllLogisticRegressionClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@class LNKRegularizationConfiguration;

/// For logistic regression classifiers, the only supported algorithm is L-BFGS.
/// Predicted values are of type LNKClass.
@interface LNKOneVsAllLogisticRegressionClassifier : LNKClassifier

@property (nonatomic, nullable, retain) LNKRegularizationConfiguration *regularizationConfiguration;

@end

NS_ASSUME_NONNULL_END
