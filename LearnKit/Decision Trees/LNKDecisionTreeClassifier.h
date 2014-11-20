//
//  LNKDecisionTreeClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// A decision tree classifier for matrices of discrete values.
/// The optimization algorithm is ignored and can be `nil`.
/// The value types of the matrix must be registered prior to training.
/// Output labels must be specified in form of classes.
@interface LNKDecisionTreeClassifier : LNKClassifier

- (void)registerBooleanValueForColumnAtIndex:(LNKSize)columnIndex;
- (void)registerCategoricalValues:(LNKSize)valueCount forColumnAtIndex:(LNKSize)columnIndex;

@end
