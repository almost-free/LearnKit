//
//  LNKNaiveBayesClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// The optimization algorithm for Naive Bayes classifiers is ignored and can be `nil`.
/// All the features of the matrix should be categorical and represented as integers.
/// Conditional independence is assumed.
/// Predicted values are of type LNKClass.
@interface LNKNaiveBayesClassifier : LNKClassifier

/// Prior to training, all possible value types must be registered for each column.
- (void)registerValues:(NSArray *)values forColumn:(LNKSize)columnIndex;

@end
