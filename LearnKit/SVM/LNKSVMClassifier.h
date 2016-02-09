//
//  LNKSVMClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// A hinge-loss SVM classifier for binary classification.
/// The only supported optimization algorithm is stochastic gradient descent.
/// An SVM may perform better if the input matrix (and feature vectors) are normalized.
/// The output classes must currently be -1 and 1.
@interface LNKSVMClassifier : LNKClassifier
@end
