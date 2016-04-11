//
//  LNKLogisticRegressionClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

/// For logistic regression classifiers, the only supported algorithm is L-BFGS.
/// Two classes are defined by default, and predicted values are of type NSNumber / LNKFloat.
/// A bias column is added to the matrix automatically.
@interface LNKLogisticRegressionClassifier : LNKClassifier
@end

NS_ASSUME_NONNULL_END
