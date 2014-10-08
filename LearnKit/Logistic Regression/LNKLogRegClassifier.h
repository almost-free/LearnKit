//
//  LNKLogRegClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// For logistic regression classifiers, the only supported algorithm is L-BFGS.
/// Two classes are defined by default, and predicted values are of type NSNumber / LNKFloat.
@interface LNKLogRegClassifier : LNKClassifier

@end
