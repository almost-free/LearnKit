//
//  LNKSVMClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@class LNKRegularizationConfiguration;

/// A hinge-loss SVM classifier for binary classification.
/// The only supported optimization algorithm is stochastic gradient descent.
/// An SVM may perform better if the input matrix (and feature vectors) are normalized.
/// The output classes must currently be -1 and 1.
/// A bias column is added to the matrix automatically.
@interface LNKSVMClassifier : LNKClassifier

@property (nonatomic, nullable, retain) LNKRegularizationConfiguration *regularizationConfiguration;

@end

NS_ASSUME_NONNULL_END
