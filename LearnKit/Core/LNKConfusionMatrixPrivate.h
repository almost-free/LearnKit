//
//  LNKConfusionMatrixPrivate.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKConfusionMatrix.h"

@interface LNKConfusionMatrix (Private)

- (void)_incrementFrequencyForTrueClass:(LNKClass *)trueClass predictedClass:(LNKClass *)predictedClass;

@end
