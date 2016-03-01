//
//  _LNKNeuralNetClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKNeuralNetClassifierAC.h"

#import "LNKAccelerate.h"
#import "LNKClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKMatrixPrivate.h"
#import "LNKMemoryBufferManager.h"
#import "LNKNeuralNetClassifierPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@interface _LNKNeuralNetClassifierAC () <LNKOptimizationAlgorithmDelegate>

@end

@implementation _LNKNeuralNetClassifierAC {
	LNKMatrix *_shuffledMatrix;
}

- (void)dealloc {
	[_shuffledMatrix release];
	[super dealloc];
}

- (void)_computeGradientForExamplesInRange:(LNKRange)range delta:(LNKFloat ***)deltas {
	NSParameterAssert(range.length);
	NSParameterAssert(deltas);
	
	LNKMatrix *matrix = self.shuffleMatrixOnEachIteration ? _shuffledMatrix : self.matrix;
	LNKClasses *classes = self.classes;
	const LNKSize classesCount = classes.count;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	const LNKSize layerCount = self.layerCount;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKSize *unitsInThetaVector = malloc(thetaVectorCount * sizeof(LNKSize)); // Cache
	*deltas = malloc(thetaVectorCount * sizeof(LNKFloat *));
	LNKFloat **deltasValue = *deltas;
	LNKFloat **gradients = malloc(thetaVectorCount * sizeof(LNKFloat *));
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		unitsInThetaVector[i] = [self _unitsInThetaVectorAtIndex:i];
		deltasValue[i] = LNKFloatCalloc(unitsInThetaVector[i]); // Initially zero
		gradients[i] = LNKFloatAlloc(unitsInThetaVector[i]);
	}
	
	LNKFloat **activations = malloc(layerCount * sizeof(LNKFloat *));
	
	// Accumulate the deltas through all the examples.
	for (LNKSize m = range.location; m < range.location + range.length; m++) {
		const LNKFloat *featureVector = _ROW_IN_MATRIX_BUFFER(m);
		
		// First predict the output, then use backpropagation to find weight gradients.
		[self _feedForwardFeatureVector:LNKVectorMakeUnsafe(featureVector, columnCount) activations:activations outputVector:NULL];
		
		// Calculate the error for the output layer (the last array of activations).
		LNKFloat *outputError = LNKFloatAllocAndCopy(activations[layerCount-1], classesCount);
		
		// In this case, the error signal s_outputLayer = a_outputLayer - y
		const LNKSize classIndex = [classes indexForClass:[LNKClass classWithUnsignedInteger:outputVector[m]]];
		outputError[classIndex] -= 1; // This is the row where y = 1
		
		[self _runBackpropogationForActivations:activations outputError:outputError gradients:gradients];
		
		// Accumulate the gradients.
		for (LNKSize i = 0; i < layerCount; i++) {
			if (i < layerCount - 1) // ignore the input layer
				LNK_vadd(deltasValue[i], UNIT_STRIDE, gradients[i], UNIT_STRIDE, deltasValue[i], UNIT_STRIDE, unitsInThetaVector[i]);
			
			free(activations[i]);
		}
	}
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		free(gradients[i]);
	}
	
	free(gradients);
	free(unitsInThetaVector);
	free(activations);
}

- (void)_runBackpropogationForActivations:(LNKFloat **)activations
							  outputError:(LNKFloat *)outputError
								gradients:(LNKFloat **)gradients {
	
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	const LNKSize layerPrior = 1;
	
	for (LNKSize layerIndex = thetaVectorCount; layerIndex >= 1; layerIndex--) {
		if (layerIndex == thetaVectorCount) {
			LNKSize rows, columns;
			[self _getDimensionsOfLayerAtIndex:layerIndex - layerPrior rows:&rows columns:&columns]; // These weights map from layer-1 to the outputLayer
			
			// The term that gets accumulated is s_outputLayer * a_layer-1
			LNK_mmul(outputError, UNIT_STRIDE, activations[layerIndex - layerPrior], UNIT_STRIDE, gradients[layerIndex - layerPrior], UNIT_STRIDE, rows, columns, 1);
		}
		else {
			// Propagate the error vector going from layer -> layer-1.
			LNKSize rows, columns;
			const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:layerIndex rows:&rows columns:&columns];
			const LNKSize columnsIgnoringBias = columns - 1;
			
			// s_i = theta * s_i+1
			LNKFloat *errorVector = LNKFloatAlloc(columns);
			LNK_mmul(thetaVector, UNIT_STRIDE, outputError, UNIT_STRIDE, errorVector, UNIT_STRIDE, columns, 1, rows);
			
			free(outputError);
			outputError = errorVector;
			
			LNKFloat *errorVectorIgnoringBias = errorVector + 1;
			LNKFloat *activationGradient = LNKMemoryBufferManagerAllocBlock(memoryManager, columnsIgnoringBias);
			LNKNeuralNetLayer *layer = [self layerAtIndex:layerIndex];
			layer.activationGradientFunction(activations[layerIndex] + 1 /* ignore bias unit */, activationGradient, columnsIgnoringBias);
			
			// Multiply the error vector by the derivative of the activation function.
			// s_i = s_i * g'(i)
			LNK_vmul(errorVectorIgnoringBias, UNIT_STRIDE, activationGradient, UNIT_STRIDE, errorVectorIgnoringBias, UNIT_STRIDE, columnsIgnoringBias);
			LNKMemoryBufferManagerFreeBlock(memoryManager, activationGradient, columnsIgnoringBias);
			
			LNKSize previousRows, previousColumns;
			[self _getDimensionsOfLayerAtIndex:layerIndex - layerPrior rows:&previousRows columns:&previousColumns];
			
			if (previousRows != columnsIgnoringBias)
				[NSException raise:NSGenericException format:@"The transition to layer %lld is invalid due to incompatible matrix sizes", layerIndex];
			
			// Error term = s_i * a_i-1
			const LNKFloat *activationVector = activations[layerIndex - layerPrior];
			LNK_mmul(errorVectorIgnoringBias, UNIT_STRIDE, activationVector, UNIT_STRIDE, gradients[layerIndex - layerPrior], UNIT_STRIDE, columnsIgnoringBias, previousColumns, 1);
		}
	}
	
	free(outputError);
}

- (void)computeGradientForOptimizationAlgorithm:(LNKFloat *)gradient inRange:(LNKRange)range {
	NSParameterAssert(gradient);
	
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	
	LNKSize *unitsInThetaVector = malloc(thetaVectorCount * sizeof(LNKSize)); // Cache
	LNKFloat **globalDeltas = malloc(thetaVectorCount * sizeof(LNKFloat *));
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		unitsInThetaVector[i] = [self _unitsInThetaVectorAtIndex:i];
		globalDeltas[i] = LNKFloatCalloc(unitsInThetaVector[i]); // Initially zero
	}
	
	const NSUInteger processorCount = _parallelProcessorCount();
	LNKFloat ***results = malloc(processorCount * sizeof(LNKFloat **));
	
	// We can parallelize running FP-BP on examples.
	[self _parallelReduceExamplesInRange:range worker:^(LNKRange innerRange, NSUInteger index) {
		[self _computeGradientForExamplesInRange:innerRange delta:&results[index]];
	}];
	
	for (NSUInteger p = 0; p < processorCount; p++) {
		for (LNKSize i = 0; i < thetaVectorCount; i++) {
			LNK_vadd(globalDeltas[i], UNIT_STRIDE, results[p][i], UNIT_STRIDE, globalDeltas[i], UNIT_STRIDE, unitsInThetaVector[i]);
			free(results[p][i]);
		}
		
		free(results[p]);
	}
	
	free(results);
	
	// Need the 1/m factor.
	const LNKFloat m = (LNKFloat)range.length;
	LNKSize gradientOffset = 0;
	
	LNKOptimizationAlgorithmRegularizable *algorithm = self.algorithm;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		LNK_vsdiv(globalDeltas[i], UNIT_STRIDE, &m, globalDeltas[i], UNIT_STRIDE, unitsInThetaVector[i]);
		
		// (Optional) regularization.
		if (algorithm.regularizationEnabled) {
			// Delta += lambda / m * Theta
			LNKSize rows, columns;
			const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:i rows:&rows columns:&columns];

			LNKFloat *thetaVectorCopy = LNKMemoryBufferManagerAllocBlock(memoryManager, rows * columns);
			LNKFloatCopy(thetaVectorCopy, thetaVector, rows * columns);

			// Ignore weights corresponding to bias units (first column).
			LNK_vclr(thetaVectorCopy, columns, rows);

			const LNKFloat factor = algorithm.lambda / m;
			LNK_vsmul(thetaVectorCopy, UNIT_STRIDE, &factor, thetaVectorCopy, UNIT_STRIDE, rows * columns);

			LNK_vadd(thetaVectorCopy, UNIT_STRIDE, globalDeltas[i], UNIT_STRIDE, globalDeltas[i], UNIT_STRIDE, rows * columns);
			LNKMemoryBufferManagerFreeBlock(memoryManager, thetaVectorCopy, rows * columns);
		}
		
		// Pack the gradient.
		LNKFloatCopy(gradient + gradientOffset, globalDeltas[i], unitsInThetaVector[i]);
		gradientOffset += unitsInThetaVector[i];
	}
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		free(globalDeltas[i]);
	}
	
	free(globalDeltas);
	free(unitsInThetaVector);
}

- (void)_initializeRandomThetaVectors {
	const LNKSize hiddenLayerUnitCount = [self hiddenLayerAtIndex:0].unitCount;
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	
	LNKSize columnCount[thetaVectorCount];
	LNKSize rowCount[thetaVectorCount];
	
	// input layer -> first hidden layer
	columnCount[0] = self.matrix.columnCount /* already includes the ones column */;
	rowCount[0] = hiddenLayerUnitCount;
	
	// last hidden layer -> output layer
	columnCount[thetaVectorCount-1] = hiddenLayerUnitCount + 1 /* bias unit */;
	rowCount[thetaVectorCount-1] = self.classes.count;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		[self _setRandomThetaVectorForLayerAtIndex:i rows:rowCount[i] columns:columnCount[i]];
	}
}

- (void)train {
	if (self.classes.count < 3)
		[NSException raise:NSGenericException format:@"Neural networks should be trained with at least three output classes"];
	
	[self _initializeRandomThetaVectors];
	
	const LNKSize totalUnitCount = [self _totalUnitCount];
	
	// Minimizing theta
	LNKFloat *thetaUnrolled = LNKFloatAlloc(totalUnitCount);
	[self _copyUnrolledThetaVectorIntoVector:thetaUnrolled];
	
	[self.algorithm runWithParameterVector:LNKVectorMakeUnsafe(thetaUnrolled, totalUnitCount) exampleCount:self.matrix.rowCount delegate:self];
	
	free(thetaUnrolled);
}

- (void)optimizationAlgorithmWillBeginIteration {
	if (!self.shuffleMatrixOnEachIteration)
		return;
	
	if (_shuffledMatrix)
		[_shuffledMatrix release];
	
	_shuffledMatrix = [self.matrix copyShuffledMatrix];
}

- (void)optimizationAlgorithmWillBeginWithInputVector:(const LNKFloat *)inputVector {
	// Unroll the inputVector into our Theta vectors.
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	LNKSize offset = 0;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		[self _updateThetaVector:inputVector + offset atIndex:i];
		offset += [self _unitsInThetaVectorAtIndex:i];
	}
}

/// Only one of `activations` or `outOutputVector` can be retrieved.
/// `outOutputVector`/members of `activations` must be freed by the caller.
- (void)_feedForwardFeatureVector:(LNKVector)featureVector activations:(LNKFloat **)activations outputVector:(LNKFloat **)outOutputVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);
	NSParameterAssert((activations && !outOutputVector) || (!activations && outOutputVector));
	
	if (activations) // The activation values for the input layer are just the original feature vectors.
		activations[0] = LNKFloatAllocAndCopy(featureVector.data, featureVector.length);
	
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	const LNKFloat *currentInputLayer = featureVector.data;
	LNKSize currentInputLayerLength = self.matrix.columnCount;
	
	for (LNKSize layerIndex = 1 /* ignore input layer */; layerIndex <= thetaVectorCount; layerIndex++) {
		LNKNeuralNetLayer *layer = [self layerAtIndex:layerIndex];
		
		LNKSize rows, columns;
		const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:layerIndex-1 rows:&rows columns:&columns];
		
		if (currentInputLayerLength != columns)
			[NSException raise:NSGenericException format:@"The transition to layer %lld is invalid due to incompatible theta matrix sizes", layerIndex];
		
		// We don't need a bias unit when prediciting outputs.
		const BOOL shouldAddBiasUnit = layerIndex != thetaVectorCount;
		const LNKSize biasUnitOffset = shouldAddBiasUnit ? 1 : 0;
		const LNKSize actualOutputVectorLength = rows + biasUnitOffset;
		
		// Perform linear combination: featureVector . thetaVector
		// In anticipation of the bias unit we may need to add, allocate space for one more element.
		LNKFloat *outputVector = LNKFloatAlloc(actualOutputVectorLength);
		LNK_mmul(thetaVector, UNIT_STRIDE, currentInputLayer, UNIT_STRIDE, outputVector + biasUnitOffset, UNIT_STRIDE, rows, 1, columns);
		
		if (shouldAddBiasUnit)
			outputVector[0] = 1;
		
		if (currentInputLayer != featureVector.data) {
			NSCAssert(layerIndex >= 1, @"The layer index must be greater than 0");
			free((void *)currentInputLayer);
		}
		
		// Apply the layer's activation function:
		layer.activationFunction(outputVector + biasUnitOffset, rows);
		
		if (activations) {
			activations[layerIndex] = LNKFloatAllocAndCopy(outputVector, actualOutputVectorLength);
			// We ignore the bias unit.
		}
		
		currentInputLayer = outputVector;
		currentInputLayerLength = actualOutputVectorLength;
	}
	
	if (currentInputLayerLength != self.classes.count)
		[NSException raise:NSGenericException format:@"Every class must be given an output"];
	
	if (outOutputVector)
		*outOutputVector = (LNKFloat *)currentInputLayer;
}

- (void)_predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);
	
	LNKFloat *outputLayer;
	[self _feedForwardFeatureVector:featureVector activations:NULL outputVector:&outputLayer];
	
	LNKSize index = 0;
	for (LNKClass *class in self.classes) {
		[self _didPredictProbability:outputLayer[index++] forClass:class];
	}
	
	free(outputLayer);
}

- (BOOL)shuffleMatrixOnEachIteration {
	return [self.algorithm isKindOfClass:[LNKOptimizationAlgorithmStochasticGradientDescent class]];
}

- (LNKFloat)_evaluateCostFunctionForExamplesInRange:(LNKRange)range {
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	LNKMatrix *matrix = self.shuffleMatrixOnEachIteration ? _shuffledMatrix : self.matrix;
	LNKClasses *classes = self.classes;
	const LNKSize classesCount = self.classes.count;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *classOutputVector = matrix.outputVector;
	const int classesCountInt = (int)classesCount;
	const LNKFloat one = 1;
	
	LNKFloat *logVector = LNKMemoryBufferManagerAllocBlock(memoryManager, classesCount);
	LNKFloat *outputVector = LNKMemoryBufferManagerAllocBlock(memoryManager, classesCount);
	
	LNKFloat J = 0;
	
	for (LNKSize m = range.location; m < range.location + range.length; m++) {
		const LNKFloat *featureVector = _ROW_IN_MATRIX_BUFFER(m);
		LNKFloat *outputLayer;
		[self _feedForwardFeatureVector:LNKVectorMakeUnsafe(featureVector, columnCount) activations:NULL outputVector:&outputLayer];
		
		// Optimize for true positives and negatives for all classes.
		// -y log(h) - (1 - y) log(1 - h)
		LNK_vlog(logVector, outputLayer, &classesCountInt);
		
		LNK_vclr(outputVector, UNIT_STRIDE, classesCount);
		const LNKSize index = [classes indexForClass:[LNKClass classWithUnsignedInteger:classOutputVector[m]]];
		
		// The output vector y will only hold a 1 for the example corresponding to the true class.
		outputVector[index] = -1; // Note -y log(h)...
		
		LNKFloat sum;
		LNK_dotpr(outputVector, UNIT_STRIDE, logVector, UNIT_STRIDE, &sum, classesCount);
		J += sum;
		
		// When subtracting 1 - y, the output vector will only hold a 0 for the example corresponding to the true class.
		LNK_vfill(&one, outputVector, UNIT_STRIDE, classesCount);
		outputVector[index] = 0;
		
		LNK_vneg(outputLayer, UNIT_STRIDE, logVector, UNIT_STRIDE, classesCount);
		free(outputLayer);
		
		LNK_vsadd(logVector, UNIT_STRIDE, &one, logVector, UNIT_STRIDE, classesCount);
		LNK_vlog(logVector, logVector, &classesCountInt);
		
		LNK_dotpr(logVector, UNIT_STRIDE, outputVector, UNIT_STRIDE, &sum, classesCount);
		J -= sum;
	}
	
	LNKMemoryBufferManagerFreeBlock(memoryManager, logVector, classesCount);
	LNKMemoryBufferManagerFreeBlock(memoryManager, outputVector, classesCount);
	
	return J;
}

- (LNKFloat)costForOptimizationAlgorithm {
	return [self _evaluateCostFunction];
}

- (LNKFloat)_evaluateCostFunction {
	const LNKSize exampleCount = self.matrix.rowCount;
	const NSUInteger processorCount = _parallelProcessorCount();
	LNKFloat *results = LNKFloatAlloc(processorCount);
	
	// We can parallelize running feed-forward on examples.
	[self _parallelReduceExamplesInRange:LNKRangeMake(0, exampleCount) worker:^(LNKRange range, NSUInteger index) {
		results[index] = [self _evaluateCostFunctionForExamplesInRange:range];
	}];
	
	LNKFloat J;
	LNK_vsum(results, UNIT_STRIDE, &J, processorCount);
	free(results);
	
	// Finally take into account the 1/m factor.
	J /= exampleCount;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
	if (algorithm.regularizationEnabled) {
		const LNKSize thetaVectorCount = [self _thetaVectorCount];
		LNKFloat regularizationCost = 0;
		
		// lambda / (2m) * sum(pow(theta, 2))
		// Note we exclude weights corresponding to the bias.
		for (LNKSize i = 0; i < thetaVectorCount; i++) {
			LNKSize rows, columns;
			const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:i rows:&rows columns:&columns];
			
			// In column-major order, the first row corresponds to bias units which should not be regularized.
			LNKFloat result;
			LNK_dotpr(thetaVector + rows, UNIT_STRIDE, thetaVector + rows, UNIT_STRIDE, &result, (columns-1) * rows);
			
			regularizationCost += result;
		}
		
		J += regularizationCost * algorithm.lambda / (2 * exampleCount);
	}
	
	return J;
}

static inline NSUInteger _parallelProcessorCount() {
	return [NSProcessInfo processInfo].processorCount;
}

- (void)_parallelReduceExamplesInRange:(LNKRange)exampleRange worker:(void(^)(LNKRange range, NSUInteger index))worker {
	dispatch_queue_t queue = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
	dispatch_group_t group = dispatch_group_create();
	
	const NSUInteger processorCount = _parallelProcessorCount();
	const LNKSize exampleCount = exampleRange.length;
	const LNKSize workgroupSize = exampleCount / processorCount;
	
	for (NSUInteger p = 0; p < processorCount; p++) {
		const LNKSize location = p * workgroupSize + exampleRange.location;
		const LNKSize length = p == processorCount - 1 ? exampleCount - p * workgroupSize : workgroupSize;
		LNKRange range = LNKRangeMake(location, length);
		
		dispatch_group_async(group, queue, ^{
			worker(range, p);
		});
	}
	
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	dispatch_release(group);
	dispatch_release(queue);
}

@end
