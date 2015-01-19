//
//  LNKNeuralNetClassifierPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKNeuralNetClassifier.h"

/// Adds support for multiple Theta vectors as used in neural networks.
@interface LNKNeuralNetClassifier (Private)

- (LNKSize)_thetaVectorCount;
- (LNKSize)_totalUnitCount;
- (LNKSize)_unitsInThetaVectorAtIndex:(NSUInteger)index;

/// The Theta vector should include the bias unit and it should be counted toward the input length.
/// By matrix should be laid out in row-major order. Set transpose if it is not.
- (void)_setThetaVector:(const LNKFloat *)thetaVector transpose:(BOOL)transpose forLayerAtIndex:(LNKSize)index rows:(LNKSize)rows columns:(LNKSize)columns;
- (void)_setRandomThetaVectorForLayerAtIndex:(LNKSize)index rows:(LNKSize)rows columns:(LNKSize)columns;
- (LNKFloat *)_thetaVectorForLayerAtIndex:(LNKSize)index rows:(LNKSize *)outRows columns:(LNKSize *)outColumns NS_RETURNS_INNER_POINTER;
- (void)_getDimensionsOfLayerAtIndex:(LNKSize)index rows:(LNKSize *)outRows columns:(LNKSize *)outColumns;

/// `unrolledThetaVector` should be of totalUnitCount length.
- (void)_copyUnrolledThetaVectorIntoVector:(LNKFloat *)unrolledThetaVector;

/// The new Theta vector should be laid out in row-major order.
- (void)_updateThetaVector:(const LNKFloat *)thetaVector atIndex:(LNKSize)index;

@end
