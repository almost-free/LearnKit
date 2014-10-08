//
//  LNKNeuralNetClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// For neural network classifiers, the only supported algorithm is CG.
/// Predicted values are of type LNKClass.
@interface LNKNeuralNetClassifier : LNKClassifier

#warning TODO: assert correct values
#warning TODO: better way of specifying classes
// Each neural network has an input layer whose size is equal to the design matrix's feature count,
// an output layer whose size is equal to the number of classes, and at least 1 hidden layer.
// Design matrices used with neural network classifiers should have a bias column.

/// The default number of hidden layers is 1.
@property (nonatomic) LNKSize hiddenLayerCount;

/// The number of units in each hidden layer (excluding the bias unit).
@property (nonatomic) LNKSize hiddenLayerUnitCount;

@end
