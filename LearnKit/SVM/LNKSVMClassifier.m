//
//  LNKSVMClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKSVMClassifier.h"

#import "LNKAccelerate.h"
#import "LNKClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKSVMClassifier {
	LNKFloat *_theta;
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmStochasticGradientDescent class] ];
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [self class];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	if (classes.count != 2)
		[NSException raise:NSInvalidArgumentException format:@"Two output classes must be specified"];
	
	if (!(self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:classes]))
		return nil;
	
	_theta = LNKFloatCalloc(matrix.columnCount);
	
	return self;
}

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmStochasticGradientDescent class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmStochasticGradientDescent *algorithm = self.algorithm;
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize epochCount = algorithm.iterationCount;
	const LNKSize stepCount = algorithm.stepCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *outputVector = matrix.outputVector;
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	
	// This simplifies some math.
	const LNKFloat lambda = regularizationEnabled ? algorithm.lambda : 0;
	
	LNKFloat *workgroupCC = LNKFloatAlloc(columnCount);
	LNKFloat *workgroupCC2 = LNKFloatAlloc(columnCount);
	
	id <LNKAlpha> alphaBox = algorithm.alpha;
	const BOOL alphaIsDecaying = [alphaBox isKindOfClass:[LNKDecayingAlpha class]];
	LNKFloat alpha = alphaIsDecaying ? 0 : [(LNKFixedAlpha *)alphaBox value];
	
	for (LNKSize epoch = 0; epoch < epochCount; epoch++) {
		if (alphaIsDecaying)
			alpha = [(LNKDecayingAlpha *)alphaBox function](epoch);
		
		for (LNKSize step = 0; step < stepCount; step++) {
			const LNKSize index = arc4random_uniform((uint32_t)matrix.rowCount);
			const LNKFloat *row = [matrix rowAtIndex:index];
			const LNKFloat output = outputVector[index];
			
			// Gradient (if y_k (Theta . x) >= 1):
			//     Theta -= alpha * (lambda * Theta)
			// Else:
			//     Theta -= alpha * (lambda * Theta - y_k * x)
			LNKFloat inner;
			LNK_dotpr(row, UNIT_STRIDE, _theta, UNIT_STRIDE, &inner, columnCount);
			inner *= output;
			
			if (inner >= 1) {
				LNK_vsmul(_theta, UNIT_STRIDE, &lambda, workgroupCC, UNIT_STRIDE, columnCount);
			}
			else {
				LNK_vsmul(_theta, UNIT_STRIDE, &lambda, workgroupCC, UNIT_STRIDE, columnCount);
				LNK_vsmul(row, UNIT_STRIDE, &output, workgroupCC2, UNIT_STRIDE, columnCount);
				LNK_vsub(workgroupCC2, UNIT_STRIDE, workgroupCC, UNIT_STRIDE, workgroupCC, UNIT_STRIDE, columnCount);
			}
			
			LNK_vsmul(workgroupCC, UNIT_STRIDE, &alpha, workgroupCC, UNIT_STRIDE, columnCount);
			LNK_vsub(workgroupCC, UNIT_STRIDE, _theta, UNIT_STRIDE, _theta, UNIT_STRIDE, columnCount);
		}
	}
	
	free(workgroupCC);
	free(workgroupCC2);
}

- (LNKFloat)_evaluateCostFunction {
	// Hinge-loss cost function: sum over N: max(0, 1 - y_k (Theta . x + b)) + 0.5 * lambda * Theta^T Theta
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *const outputVector = matrix.outputVector;
	LNKOptimizationAlgorithmStochasticGradientDescent *const algorithm = self.algorithm;
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = regularizationEnabled ? algorithm.lambda : 0;

	LNKFloat cost = 0;

	for (LNKSize exampleIndex = 0; exampleIndex < rowCount; exampleIndex++) {
		LNKFloat y = 0;
		LNK_dotpr(_theta, UNIT_STRIDE, [matrix rowAtIndex:exampleIndex], UNIT_STRIDE, &y, columnCount);
		//TODO: include the +b term
		
		cost += MAX(0, 1 - outputVector[exampleIndex] * y);
	}

	LNKFloat thetaSquared = 0;
	LNK_dotpr(_theta, UNIT_STRIDE, _theta, UNIT_STRIDE, &thetaSquared, columnCount);
	cost += 0.5 * lambda * thetaSquared;

	return cost;
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	if (!featureVector.data)
		[NSException raise:NSInvalidArgumentException format:@"The feature vector must not be NULL"];
	
	const LNKSize columnCount = self.matrix.columnCount;
	
	if (columnCount != featureVector.length)
		[NSException raise:NSInvalidArgumentException format:@"The length of the feature vector must match the number of columns in the training matrix"];
	
	LNKFloat result;
	LNK_dotpr(featureVector.data, UNIT_STRIDE, _theta, UNIT_STRIDE, &result, columnCount);
	
	return @(result);
}

- (LNKFloat)computeClassificationAccuracyOnMatrix:(LNKMatrix *)matrix {
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKSize hits = 0;
	
	for (LNKSize m = 0; m < rowCount; m++) {
		id predictedValue = [self predictValueForFeatureVector:LNKVectorMakeUnsafe([matrix rowAtIndex:m], columnCount)];
		NSAssert([predictedValue isKindOfClass:[NSNumber class]], @"Unexpected value");
		
		const LNKFloat value = [predictedValue LNKFloatValue];
		
		if (value * outputVector[m] > 0)
			hits++;
	}
	
	return (LNKFloat)hits / rowCount;
}

- (void)dealloc {
	free(_theta);
	
	[super dealloc];
}

@end
