//
//  _LNKNeuralNetClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKNeuralNetClassifierAC.h"

#import "fmincg.h"
#import "LNKAccelerate.h"
#import "LNKClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKMatrixPrivate.h"
#import "LNKMemoryBufferManager.h"
#import "LNKNeuralNetClassifierPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKNeuralNetClassifierAC

static _LNKNeuralNetClassifierAC *tempSelf = nil;


static void _fmincg_evaluate(LNKFloat *inputVector, LNKFloat *outCost, LNKFloat *gradientVector) {
	_LNKNeuralNetClassifierAC *self = tempSelf;
	assert(self);
	assert(inputVector);
	assert(outCost);
	assert(gradientVector);
	
	// Unroll the inputVector into our Theta vectors.
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	LNKSize offset = 0;
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		[self _updateThetaVector:inputVector + offset atIndex:i];
		offset += [self _unitsInThetaVectorAtIndex:i];
	}
	
	const LNKFloat cost = [self _evaluateCostFunction];
	[self _computeGradientAndCopyToVector:gradientVector];
	
	*outCost = cost;
}

- (void)_computeGradientForExamplesInRange:(NSRange)range delta:(LNKFloat ***)deltas {
	NSParameterAssert(range.length);
	NSParameterAssert(deltas);
	
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	LNKMatrix *matrix = self.matrix;
	LNKClasses *classes = self.classes;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	const LNKFloat *outputVector = matrix.outputVector;
	const LNKSize inputLayerOffset = 1, layerPrior = 1;
	
	LNKSize *unitsInThetaVector = malloc(thetaVectorCount * sizeof(LNKSize)); // Cache
	*deltas = malloc(thetaVectorCount * sizeof(LNKFloat *));
	LNKFloat **deltasValue = *deltas;
	LNKFloat **tempBuffers = malloc(thetaVectorCount * sizeof(LNKFloat *));
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		unitsInThetaVector[i] = [self _unitsInThetaVectorAtIndex:i];
		deltasValue[i] = LNKFloatCalloc(unitsInThetaVector[i]); // Initially zero
		tempBuffers[i] = LNKFloatAlloc(unitsInThetaVector[i]);
	}
	
	const LNKSize hiddenLayerCount = self.hiddenLayerCount;
	LNKFloat **hiddenLayerActivations = malloc(hiddenLayerCount * sizeof(LNKFloat *));
	
	// Accumulate the deltas through all the examples.
	for (LNKSize m = range.location; m < NSMaxRange(range); m++) {
		const LNKFloat *featureVector = _EXAMPLE_IN_MATRIX_BUFFER(m);
		LNKFloat *currentErrorVector;
		
		// First predict the output, then propagate the error.
		[self _feedForwardFeatureVector:LNKVectorMakeUnsafe(featureVector, columnCount) hiddenLayerActivations:hiddenLayerActivations outputVector:&currentErrorVector];
		
		for (LNKSize layer = thetaVectorCount; layer >= 1; layer--) {
			if (layer == thetaVectorCount) {
				// In this case, the error signal s_outputLayer = a_outputLayer - y
				const LNKSize classIndex = [classes indexForClass:[LNKClass classWithUnsignedInteger:outputVector[m]]];
				currentErrorVector[classIndex] -= 1; // This is the row where y = 1
				
				LNKSize rows, columns;
				[self _getDimensionsOfLayerAtIndex:layer - layerPrior rows:&rows columns:&columns]; // These weights map from layer-1 to the outputLayer
				
				// The term that gets accumulated is s_outputLayer * a_layer-1
				LNK_mmul(currentErrorVector, UNIT_STRIDE, hiddenLayerActivations[layer - inputLayerOffset - layerPrior], UNIT_STRIDE, tempBuffers[layer - layerPrior], UNIT_STRIDE, rows, columns, 1);
			}
			else {
				// Propagate the error vector going from layer -> layer-1.
				LNKSize rows, columns;
				const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:layer rows:&rows columns:&columns];
				const LNKSize columnsIgnoringBias = columns - 1;
				
				// s_i = theta * s_i+1
				LNKFloat *errorVector = LNKFloatAlloc(columns);
				LNK_mmul(thetaVector, UNIT_STRIDE, currentErrorVector, UNIT_STRIDE, errorVector, UNIT_STRIDE, columns, 1, rows);
				
				free(currentErrorVector);
				currentErrorVector = errorVector;
				
				LNKFloat *errorVectorIgnoringBias = errorVector + 1;
				LNKFloat *sigmoidGradient = LNKMemoryBufferManagerAllocBlock(memoryManager, columnsIgnoringBias);
				LNK_vsigmoidgrad(hiddenLayerActivations[layer - layerPrior] + 1 /* ignore bias unit */, sigmoidGradient, columnsIgnoringBias);
				
				// Multiply the error vector by the sigmoid gradient.
				// s_i * g'(i)
				LNK_vmul(errorVectorIgnoringBias, UNIT_STRIDE, sigmoidGradient, UNIT_STRIDE, errorVectorIgnoringBias, UNIT_STRIDE, columnsIgnoringBias);
				LNKMemoryBufferManagerFreeBlock(memoryManager, sigmoidGradient, columnsIgnoringBias);
				
				LNKSize previousRows, previousColumns;
				[self _getDimensionsOfLayerAtIndex:layer - layerPrior rows:&previousRows columns:&previousColumns];
				
				if (previousRows != columnsIgnoringBias)
					[NSException raise:NSGenericException format:@"The transition to layer %lld is invalid due to incompatible matrix sizes", layer];
				
				LNK_mmul(errorVectorIgnoringBias, UNIT_STRIDE, featureVector, UNIT_STRIDE, tempBuffers[layer - layerPrior], UNIT_STRIDE, columnsIgnoringBias, previousColumns, 1);
			}
		}
		
		free(currentErrorVector);
		
		// Accumulate deltas.
		for (LNKSize i = 0; i < thetaVectorCount; i++) {
			LNK_vadd(deltasValue[i], UNIT_STRIDE, tempBuffers[i], UNIT_STRIDE, deltasValue[i], UNIT_STRIDE, unitsInThetaVector[i]);
			free(hiddenLayerActivations[i]);
		}
	}
	
	for (LNKSize i = 0; i < thetaVectorCount; i++) {
		free(tempBuffers[i]);
	}
	
	free(tempBuffers);
	free(unitsInThetaVector);
	free(hiddenLayerActivations);
}

- (void)_computeGradientAndCopyToVector:(LNKFloat *)gradient {
	NSParameterAssert(gradient);
	
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize exampleCount = matrix.exampleCount;
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
	[self _parallelReduceExamples:exampleCount worker:^(NSRange range, NSUInteger index) {
		[self _computeGradientForExamplesInRange:range delta:&results[index]];
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
	const LNKFloat m = (LNKFloat)exampleCount;
	LNKSize gradientOffset = 0;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
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
	const LNKSize hiddenLayerUnitCount = self.hiddenLayerUnitCount;
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
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
	tempSelf = self;
	
#ifdef DEBUG
	int result = fmincg(_fmincg_evaluate, thetaUnrolled, (int)totalUnitCount, (int)algorithm.iterationCount);
	NSAssert(result == 0 || result == 1, @"Could not minimize the function");
#else
	fmincg(_fmincg_evaluate, thetaUnrolled, (int)totalUnitCount, (int)algorithm.iterationCount);
#endif
	
	free(thetaUnrolled);
}

/// `outOutputVector` and `outActivations` (and its members) must be freed by the caller.
- (void)_feedForwardFeatureVector:(LNKVector)featureVector hiddenLayerActivations:(LNKFloat **)activations outputVector:(LNKFloat **)outOutputVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);
	NSParameterAssert(outOutputVector);
	
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	const LNKSize thetaVectorCount = [self _thetaVectorCount];
	const LNKFloat *currentInputLayer = featureVector.data;
	LNKSize currentInputLayerLength = self.matrix.columnCount;
	
	for (LNKSize layer = 0; layer < thetaVectorCount; layer++) {
		LNKSize rows, columns;
		const LNKFloat *thetaVector = [self _thetaVectorForLayerAtIndex:layer rows:&rows columns:&columns];
		
		if (currentInputLayerLength != columns)
			[NSException raise:NSGenericException format:@"The transition to layer %lld is invalid due to incompatible theta matrix sizes", layer];
		
		LNKFloat *transposedThetaVector = LNKMemoryBufferManagerAllocBlock(memoryManager, rows * columns);
		LNK_mtrans(thetaVector, UNIT_STRIDE, transposedThetaVector, UNIT_STRIDE, columns, rows);
		
		// We don't need a bias unit when prediciting outputs.
		const BOOL shouldAddBiasUnit = layer < thetaVectorCount-1;
		const LNKSize biasUnitOffset = shouldAddBiasUnit ? 1 : 0;
		const LNKSize actualOutputVectorLength = rows + biasUnitOffset;
		
		// sigmoid(featureVector . thetaVector) -> next layer
		// In anticipation of the bias unit we may need to add, allocate space for one more element.
		LNKFloat *outputVector = LNKFloatAlloc(actualOutputVectorLength);
		LNK_mmul(currentInputLayer, UNIT_STRIDE, transposedThetaVector, UNIT_STRIDE, outputVector + biasUnitOffset, UNIT_STRIDE, 1, rows, columns);
		LNKMemoryBufferManagerFreeBlock(memoryManager, transposedThetaVector, rows * columns);
		
		if (shouldAddBiasUnit)
			outputVector[0] = 1;
		
		if (currentInputLayer != featureVector.data) {
			assert(layer >= 1);
			free((void *)currentInputLayer);
		}
		
		LNK_vsigmoid(outputVector + biasUnitOffset, rows);
		
		if (activations) {
			activations[layer] = LNKFloatAlloc(rows);
			
			// We ignore the bias unit.
			LNKFloatCopy(activations[layer], outputVector, actualOutputVectorLength);
		}
		
		currentInputLayer = outputVector;
		currentInputLayerLength = actualOutputVectorLength;
	}
	
	if (currentInputLayerLength != self.classes.count)
		[NSException raise:NSGenericException format:@"Every class must be given an output"];
	
	*outOutputVector = (LNKFloat *)currentInputLayer;
}

- (void)_predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);
	
	LNKFloat *outputLayer;
	[self _feedForwardFeatureVector:featureVector hiddenLayerActivations:NULL outputVector:&outputLayer];
	
	LNKSize index = 0;
	for (LNKClass *class in self.classes) {
		[self _didPredictProbability:outputLayer[index++] forClass:class];
	}
	
	free(outputLayer);
}

- (LNKFloat)_evaluateCostFunctionForExamplesInRange:(NSRange)range {
	LNKMemoryBufferManagerRef memoryManager = LNKGetCurrentMemoryBufferManager();
	
	LNKMatrix *matrix = self.matrix;
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
	
	for (LNKSize m = range.location; m < NSMaxRange(range); m++) {
		const LNKFloat *featureVector = _EXAMPLE_IN_MATRIX_BUFFER(m);
		LNKFloat *outputLayer;
		[self _feedForwardFeatureVector:LNKVectorMakeUnsafe(featureVector, columnCount) hiddenLayerActivations:NULL outputVector:&outputLayer];
		
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

- (LNKFloat)_evaluateCostFunction {
	const LNKSize exampleCount = self.matrix.exampleCount;
	const NSUInteger processorCount = _parallelProcessorCount();
	LNKFloat *results = LNKFloatAlloc(processorCount);
	
	// We can parallelize running feed-forward on examples.
	[self _parallelReduceExamples:exampleCount worker:^(NSRange range, NSUInteger index) {
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

- (void)_parallelReduceExamples:(LNKSize)exampleCount worker:(void(^)(NSRange range, NSUInteger index))worker {
	dispatch_queue_t queue = dispatch_queue_create(NULL, DISPATCH_QUEUE_CONCURRENT);
	dispatch_group_t group = dispatch_group_create();
	
	const NSUInteger processorCount = _parallelProcessorCount();
	const LNKSize workgroupSize = exampleCount / processorCount;
	
	for (NSUInteger p = 0; p < processorCount; p++) {
		NSRange range;
		range.location = p * workgroupSize;
		range.length = p == processorCount - 1 ? exampleCount - p * workgroupSize : workgroupSize;
		
		dispatch_group_async(group, queue, ^{
			worker(range, p);
		});
	}
	
	dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
	dispatch_release(group);
	dispatch_release(queue);
}

@end
