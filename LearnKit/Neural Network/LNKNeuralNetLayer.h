//
//  LNKNeuralNetLayer.h
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import "LNKClasses.h"

@interface LNKNeuralNetLayer : NSObject

/// The unit count should exclude the bias unit.
- (instancetype)initWithUnitCount:(LNKSize)unitCount NS_DESIGNATED_INITIALIZER;

@property (nonatomic, readonly) LNKSize unitCount;

@end


@interface LNKNeuralNetOutputLayer : LNKNeuralNetLayer

- (instancetype)initWithClasses:(LNKClasses *)classes NS_DESIGNATED_INITIALIZER;

@property (nonatomic, readonly) LNKClasses *classes;

@end
