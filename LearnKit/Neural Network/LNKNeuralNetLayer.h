//
//  LNKNeuralNetLayer.h
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface LNKNeuralNetLayer : NSObject

/// The unit count should exclude the bias unit.
- (instancetype)initWithUnitCount:(LNKSize)unitCount;

@property (nonatomic, readonly) LNKSize unitCount;

@end
