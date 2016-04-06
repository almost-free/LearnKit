//
//  LNKNeuralNetClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"
#import "LNKNeuralNetLayer.h"

NS_ASSUME_NONNULL_BEGIN

/// For neural network classifiers, the only supported algorithms are CG and stochastic gradient descent.
/// Predicted values are of type LNKClass.
/// A bias column is added to the matrix automatically.
@interface LNKNeuralNetClassifier : LNKClassifier

// Each neural network has an input layer whose size is equal to the matrix's feature count,
// an output layer whose size is equal to the number of classes, and at least one hidden layer.

/// At least one hidden layer must be specified.
/// The matrix must have a bias column.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix
			implementationType:(LNKImplementationType)implementation
		 optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm
				  hiddenLayers:(NSArray<LNKNeuralNetLayer *> *)layers
				   outputLayer:(LNKNeuralNetLayer *)outputLayer NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithMatrix:(LNKMatrix *)matrix
			implementationType:(LNKImplementationType)implementation
		 optimizationAlgorithm:(nullable id<LNKOptimizationAlgorithm>)algorithm
					   classes:(LNKClasses *)classes NS_UNAVAILABLE;

@property (nonatomic, readonly) LNKSize layerCount;
@property (nonatomic, readonly) LNKSize hiddenLayerCount;

@property (nonatomic, readonly) LNKNeuralNetLayer *inputLayer;
@property (nonatomic, readonly) LNKNeuralNetLayer *outputLayer;

- (LNKNeuralNetLayer *)layerAtIndex:(LNKSize)index;
- (LNKNeuralNetLayer *)hiddenLayerAtIndex:(LNKSize)index;

#warning TODO: assert correct values
#warning TODO: better way of specifying classes

@end

NS_ASSUME_NONNULL_END
