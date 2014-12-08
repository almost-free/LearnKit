//
//  LNKOptimizationAlgorithm.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

@protocol LNKOptimizationAlgorithm <NSObject>
@end

/// Abstract
@interface LNKOptimizationAlgorithmRegularizable : NSObject <LNKOptimizationAlgorithm>

@property (nonatomic) BOOL regularizationEnabled;
@property (nonatomic) LNKFloat lambda;

@end


@interface LNKOptimizationAlgorithmNormalEquations : NSObject <LNKOptimizationAlgorithm>
@end


@interface LNKOptimizationAlgorithmGradientDescent : LNKOptimizationAlgorithmRegularizable <LNKOptimizationAlgorithm>

- (instancetype)init NS_UNAVAILABLE;

/// Indicates gradient descent should run for a fixed number of iterations.
+ (instancetype)algorithmWithAlpha:(LNKFloat)alpha stochastic:(BOOL)stochastic iterationCount:(LNKSize)iterationCount;

/// Indicates gradient descent should run until convergence to a given threshold.
+ (instancetype)algorithmWithAlpha:(LNKFloat)alpha stochastic:(BOOL)stochastic convergenceThreshold:(LNKFloat)convergenceThreshold;

@property (nonatomic, readonly) LNKFloat alpha;
@property (nonatomic, readonly, getter=isStochastic) BOOL stochastic;

/// This value is `NSNotFound` when converging automatically.
@property (nonatomic, readonly) LNKSize iterationCount;

/// This value is `0` when using a fixed number of iterations.
@property (nonatomic, readonly) LNKFloat convergenceThreshold;

@end

@interface LNKOptimizationAlgorithmLBFGS : LNKOptimizationAlgorithmRegularizable <LNKOptimizationAlgorithm>
@end


@interface LNKOptimizationAlgorithmCG : LNKOptimizationAlgorithmRegularizable <LNKOptimizationAlgorithm>

/// Defaults to 100.
@property (nonatomic) NSUInteger iterationCount;

@end
