//
//  LNKOptimizationAlgorithm.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

@protocol LNKAlpha <NSObject>
@end

@interface LNKFixedAlpha : NSObject <LNKAlpha>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)withValue:(LNKFloat)value;

@property (nonatomic, readonly) LNKFloat value;

@end

typedef LNKFloat(^LNKDecayingAlphaFunction)(LNKSize iteration);

@interface LNKDecayingAlpha : NSObject <LNKAlpha>

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)withFunction:(LNKDecayingAlphaFunction)function;

@property (nonatomic, readonly) LNKDecayingAlphaFunction function;

@end


@protocol LNKOptimizationAlgorithmDelegate <NSObject>

- (void)optimizationAlgorithmWillBeginWithInputVector:(const LNKFloat *)inputVector;
- (LNKFloat)costForOptimizationAlgorithm;
- (void)computeGradientForOptimizationAlgorithm:(LNKFloat *)gradient inRange:(LNKRange)range;

@optional
- (void)optimizationAlgorithmWillBeginIteration;

@end

@protocol LNKOptimizationAlgorithm <NSObject>

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate;

@end


@interface LNKOptimizationAlgorithmNormalEquations : NSObject <LNKOptimizationAlgorithm>
@end


@interface LNKOptimizationAlgorithmGradientDescent : NSObject <LNKOptimizationAlgorithm>

- (instancetype)init NS_UNAVAILABLE;

/// Indicates gradient descent should run for a fixed number of iterations.
+ (instancetype)algorithmWithAlpha:(id <LNKAlpha>)alpha iterationCount:(LNKSize)iterationCount;

/// Indicates gradient descent should run until convergence to a given threshold.
+ (instancetype)algorithmWithAlpha:(id <LNKAlpha>)alpha convergenceThreshold:(LNKFloat)convergenceThreshold;

@property (nonatomic, readonly) id <LNKAlpha> alpha;

/// This value is `NSNotFound` when converging automatically.
@property (nonatomic, readonly) LNKSize iterationCount;

/// This value is `0` when using a fixed number of iterations.
@property (nonatomic, readonly) LNKFloat convergenceThreshold;

@end


@interface LNKOptimizationAlgorithmStochasticGradientDescent : LNKOptimizationAlgorithmGradientDescent

@property (nonatomic) LNKSize stepCount;

@end


@interface LNKOptimizationAlgorithmLBFGS : NSObject <LNKOptimizationAlgorithm>
@end


@interface LNKOptimizationAlgorithmCG : NSObject <LNKOptimizationAlgorithm>

/// Defaults to 100.
@property (nonatomic) NSUInteger iterationCount;

@end

NS_ASSUME_NONNULL_END
