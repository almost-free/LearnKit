//
//  LNKDecisionTreeClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

@interface LNKDecisionTreeClassifier : LNKClassifier

- (void)registerBooleanValueForColumnAtIndex:(LNKSize)columnIndex;
- (void)registerCategoricalValues:(LNKSize)valueCount forColumnAtIndex:(LNKSize)columnIndex;

@end
