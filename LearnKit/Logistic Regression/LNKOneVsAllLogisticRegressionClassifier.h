//
//  LNKOneVsAllLogisticRegressionClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// For logistic regression classifiers, the only supported algorithm is L-BFGS.
/// Predicted values are of type LNKClass.
@interface LNKOneVsAllLogisticRegressionClassifier : LNKClassifier

@end
