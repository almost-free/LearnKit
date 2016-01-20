//
//  LNKNeuralNetLayer.h
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import "LNKClasses.h"

typedef void(^LNKActivationFunction)(LNKFloat *vector, LNKSize length);
typedef void(^LNKActivationGradientFunction)(const LNKFloat *vector, LNKFloat *outVector, LNKSize length);

@interface LNKNeuralNetLayer : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/// The unit count should exclude the bias unit.
- (instancetype)initWithUnitCount:(LNKSize)unitCount NS_DESIGNATED_INITIALIZER;

/// Initializes a special 'output' layer with the given classes.
- (instancetype)initWithClasses:(LNKClasses *)classes;

@property (nonatomic, readonly) LNKSize unitCount;

@property (nonatomic, readonly, getter=isOutputLayer) BOOL outputLayer;
@property (nonatomic, readonly) LNKClasses *classes;

- (LNKActivationFunction)activationFunction;
- (LNKActivationGradientFunction)activationGradientFunction;

@end


@interface LNKNeuralNetSigmoidLayer : LNKNeuralNetLayer

@end


@interface LNKNeuralNetReLULayer : LNKNeuralNetLayer

@end
