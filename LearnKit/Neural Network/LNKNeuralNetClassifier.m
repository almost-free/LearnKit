//
//  LNKNeuralNetClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKNeuralNetClassifier.h"

#import "_LNKNeuralNetClassifierAC.h"
#import "LNKAccelerate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKNeuralNetClassifierPrivate.h"
#import "LNKPredictorPrivate.h"

typedef struct {
	LNKFloat *thetaVector;
	LNKSize rows, columns;
} ThetaVectorBucket;

@implementation LNKNeuralNetClassifier {
	ThetaVectorBucket **_thetaVectorBuckets;
	LNKSize _thetaVectorBucketCount;
	NSArray<LNKNeuralNetLayer *> *_layers;
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmCG class], [LNKOptimizationAlgorithmStochasticGradientDescent class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKNeuralNetClassifierAC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm hiddenLayers:(NSArray<LNKNeuralNetLayer *> *)hiddenLayers outputLayer:(LNKNeuralNetLayer *)outputLayer {
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Bias columns are added to matrices automatically by NN classifiers." userInfo:nil];
	}
	
	if (!hiddenLayers.count) {
		[NSException raise:NSInvalidArgumentException format:@"At least one hidden layer must be specified"];
	}
	
	if (!outputLayer.isOutputLayer) {
		[NSException raise:NSInvalidArgumentException format:@"An output layer must be specified"];
	}

	LNKMatrix *const workingMatrix = matrix.matrixByAddingBiasColumn;
	
	if (!(self = [super initWithMatrix:workingMatrix implementationType:implementation optimizationAlgorithm:algorithm classes:outputLayer.classes])) {
		return nil;
	}
	
	NSMutableArray<LNKNeuralNetLayer *> *allLayers = [[NSMutableArray alloc] init];
	
	LNKNeuralNetLayer *inputLayer = [[LNKNeuralNetLayer alloc] initWithUnitCount:workingMatrix.columnCount];
	[allLayers addObject:inputLayer];
	[inputLayer release];
	
	[allLayers addObjectsFromArray:hiddenLayers];
	[allLayers addObject:outputLayer];
	
	_layers = [allLayers copy];
	[allLayers release];
	
	return self;
}

#pragma mark - Layer Management

- (LNKSize)layerCount {
	return _layers.count;
}

- (LNKSize)hiddenLayerCount {
	return _layers.count - 2; // Offset by input and output layer
}

- (LNKNeuralNetLayer *)inputLayer {
	return (_Nonnull id)_layers.firstObject;
}

- (LNKNeuralNetLayer *)outputLayer {
	return (_Nonnull id)_layers.lastObject;
}

- (LNKNeuralNetLayer *)layerAtIndex:(LNKSize)index {
	if (index >= _layers.count)
		[NSException raise:NSInvalidArgumentException format:@"The layer index is out of bounds (%ld)", (unsigned long)_layers.count];
	
	return _layers[index];
}

- (LNKNeuralNetLayer *)hiddenLayerAtIndex:(LNKSize)index {
	if (index >= self.hiddenLayerCount)
		[NSException raise:NSInvalidArgumentException format:@"The hidden layer index is out of bounds (%lld)", self.hiddenLayerCount];
	
	return _layers[index + 1 /* offset by input layer */];
}

#pragma mark -

- (LNKSize)_totalUnitCount {
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	
	LNKSize count = 0;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		const ThetaVectorBucket *thetaVectorBucket = _thetaVectorBuckets[i];
		count += thetaVectorBucket->rows * thetaVectorBucket->columns;
	}
	
	return count;
}

- (LNKSize)_unitsInThetaVectorAtIndex:(NSUInteger)index {
	NSAssert(index < [self _thetaVectorCount], @"Out-of-bounds index");
	NSAssert(index < _thetaVectorBucketCount, @"Out-of-bounds index");
	
	const ThetaVectorBucket *thetaVectorBucket = _thetaVectorBuckets[index];
	return thetaVectorBucket->rows * thetaVectorBucket->columns;
}

- (LNKFloat *)_thetaVectorForLayerAtIndex:(LNKSize)index rows:(LNKSize *)outRows columns:(LNKSize *)outColumns {
	[self _getDimensionsOfLayerAtIndex:index rows:outRows columns:outColumns];
	return _thetaVectorBuckets[index]->thetaVector;
}

- (void)_getDimensionsOfLayerAtIndex:(LNKSize)index rows:(LNKSize *)outRows columns:(LNKSize *)outColumns {
	NSAssert(index < [self _thetaVectorCount], @"Out-of-bounds index");
	NSAssert(index < _thetaVectorBucketCount, @"Out-of-bounds index");
	
	const ThetaVectorBucket *thetaVectorBucket = _thetaVectorBuckets[index];
	
	if (outRows)
		*outRows = thetaVectorBucket->rows;
	
	if (outColumns)
		*outColumns = thetaVectorBucket->columns;
}

- (void)_createThetaVectorForLayerAtIndex:(LNKSize)index rows:(LNKSize)rows columns:(LNKSize)columns {
	NSAssert(index < [self _thetaVectorCount], @"Out-of-bounds index");
	NSAssert(rows, @"The row count should be non-zero");
	NSAssert(columns, @"The column count should be non-zero");
	
	if (!_thetaVectorBuckets) {
		_thetaVectorBucketCount = [self _thetaVectorCount];
		_thetaVectorBuckets = calloc(_thetaVectorBucketCount, sizeof(ThetaVectorBucket *));
	}
	
	_thetaVectorBuckets[index] = malloc(sizeof(ThetaVectorBucket));
	_thetaVectorBuckets[index]->rows = rows;
	_thetaVectorBuckets[index]->columns = columns;
	_thetaVectorBuckets[index]->thetaVector = LNKFloatAlloc(rows * columns);
}

- (void)_setRandomThetaVectorForLayerAtIndex:(LNKSize)index rows:(LNKSize)rows columns:(LNKSize)columns {
	[self _createThetaVectorForLayerAtIndex:index rows:rows columns:columns];
	
	const LNKFloat epsilon = 0.12;
	
	LNKFloat *thetaVector = _thetaVectorBuckets[index]->thetaVector;
	
	for (LNKSize row = 0; row < rows; row++) {
		for (LNKSize column = 0; column < columns; column++) {
			thetaVector[row * columns + column] = (((LNKFloat) arc4random_uniform(UINT32_MAX) / UINT32_MAX) - 0.5) * 2 * epsilon;
		}
	}
}

- (void)_setThetaVector:(const LNKFloat *)thetaVector transpose:(BOOL)transpose forLayerAtIndex:(LNKSize)index rows:(LNKSize)rows columns:(LNKSize)columns {
	NSParameterAssert(thetaVector);
	[self _createThetaVectorForLayerAtIndex:index rows:rows columns:columns];
	
	if (transpose)
		LNK_mtrans(thetaVector, _thetaVectorBuckets[index]->thetaVector, rows, columns);
	else
		LNKFloatCopy(_thetaVectorBuckets[index]->thetaVector, thetaVector, rows * columns);
}

- (void)_copyUnrolledThetaVectorIntoVector:(LNKFloat *)unrolledThetaVector {
	NSParameterAssert(unrolledThetaVector);
	
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	LNKSize unitOffset = 0;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		LNKSize rows, columns;
		LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:i rows:&rows columns:&columns];
		const LNKSize totalUnitCount = rows * columns;
		LNKFloatCopy(unrolledThetaVector + unitOffset, thetaVector, totalUnitCount);
		
		unitOffset += totalUnitCount;
	}
}

- (void)_updateThetaVector:(const LNKFloat *)thetaVector atIndex:(LNKSize)index {
	NSParameterAssert(thetaVector);
	NSAssert(index < [self _thetaVectorCount], @"Out-of-bounds index");
	NSAssert(index < _thetaVectorBucketCount, @"Out-of-bounds index");
	
	LNKFloatCopy(_thetaVectorBuckets[index]->thetaVector, thetaVector, _thetaVectorBuckets[index]->rows * _thetaVectorBuckets[index]->columns);
}

- (void)dealloc {
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	
	[_layers release];
	
	if (!_thetaVectorBuckets) {
		[super dealloc];
		return;
	}
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		if (_thetaVectorBuckets[i])
			free(_thetaVectorBuckets[i]->thetaVector);
	}
	
	free(_thetaVectorBuckets);
	
	[super dealloc];
}

- (LNKSize)_thetaVectorCount {
	// We only have totalLayerCount - 1 Theta vectors since each Theta vector transitions us to the next layer.
	return _layers.count - 1;
}

@end
