//
//  LNKCollaborativeFilteringPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKCollaborativeFilteringPredictor.h"

#import "LNKAccelerate.h"
#import "LNKCollaborativeFilteringPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

#import "fmincg.h"

@implementation LNKCollaborativeFilteringPredictor {
	LNKMatrix *_indicatorMatrix;
	LNKSize _featureCount;
	LNKFloat *_unrolledGradient;
}

+ (NSArray *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmCG class] ];
}

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

- (instancetype)initWithMatrix:(LNKMatrix *)outputMatrix indicatorMatrix:(LNKMatrix *)indicatorMatrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm featureCount:(NSUInteger)featureCount {
#pragma unused(implementationType)
	
	if (featureCount == 0)
		[NSException raise:NSInvalidArgumentException format:@"The feature count must be greater than 0"];
	
	if (!indicatorMatrix)
		[NSException raise:NSInvalidArgumentException format:@"The indicator matrix must not be nil"];
	
	self = [self initWithMatrix:outputMatrix optimizationAlgorithm:algorithm];
	if (self) {
		_featureCount = featureCount;
		_indicatorMatrix = [indicatorMatrix retain];
		
		const LNKSize userCount = outputMatrix.columnCount;
		const LNKSize exampleCount = outputMatrix.exampleCount;
		const LNKSize unrolledExampleCount = exampleCount + userCount;
		
		_unrolledGradient = LNKFloatAlloc(unrolledExampleCount * _featureCount);
	}
	return self;
}

- (LNKFloat)_evaluateCostFunction {
	LNKMatrix *outputMatrix = self.matrix;
	const LNKSize userCount = outputMatrix.columnCount;
	const LNKSize exampleCount = outputMatrix.exampleCount;
	const LNKFloat *dataMatrix = _unrolledGradient;
	const LNKFloat *thetaMatrix = _unrolledGradient + exampleCount * _featureCount;
	
	LNKFloat *thetaTranspose = LNKFloatAlloc(userCount * _featureCount);
	LNK_mtrans(thetaMatrix, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, _featureCount, userCount);
	
	// 1/2 * sum((((X * Theta') - Y) ^ 2) * R)
	const LNKSize resultSize = exampleCount * userCount;
	LNKFloat *result = LNKFloatAlloc(resultSize);
	LNK_mmul(dataMatrix, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, result, UNIT_STRIDE, exampleCount, userCount, _featureCount);
	
	LNK_vsub(outputMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, _indicatorMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	
	LNKFloat sum;
	LNK_vsum(result, UNIT_STRIDE, &sum, resultSize);
	free(result);
	
	LNKFloat regularizationTerm = 0;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
	if (algorithm.regularizationEnabled) {
		// Re-use the theta transpose matrix to compute the theta square.
		LNK_vmul(thetaTranspose, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, userCount * _featureCount);
		
		LNKFloat *dataSquare = LNKFloatAlloc(exampleCount * _featureCount);
		LNK_vmul(dataMatrix, UNIT_STRIDE, dataMatrix, UNIT_STRIDE, dataSquare, UNIT_STRIDE, exampleCount * _featureCount);
		
		LNKFloat thetaSum, dataSum;
		LNK_vsum(thetaTranspose, UNIT_STRIDE, &thetaSum, userCount * _featureCount);
		LNK_vsum(dataSquare, UNIT_STRIDE, &dataSum, exampleCount * _featureCount);
		free(dataSquare);
		
		// ... + lambda / 2 * (sum(Theta^2) + sum(X^2))
		regularizationTerm = algorithm.lambda / 2 * (thetaSum + dataSum);
	}
	
	free(thetaTranspose);
	
	return 0.5 * sum + regularizationTerm;
}

- (const LNKFloat *)_computeGradient {
	LNKMatrix *outputMatrix = self.matrix;
	const LNKSize userCount = outputMatrix.columnCount;
	const LNKSize exampleCount = outputMatrix.exampleCount;
	const LNKSize unrolledExampleCount = exampleCount + userCount;
	const LNKFloat *dataMatrix = _unrolledGradient;
	const LNKFloat *thetaMatrix = _unrolledGradient + exampleCount * _featureCount;
	
	LNKFloat *unrolledGradient = LNKFloatCalloc(unrolledExampleCount * _featureCount);
	
	LNKFloat *dataGradient = unrolledGradient;
	LNKFloat *thetaGradient = unrolledGradient + exampleCount * _featureCount;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = algorithm.lambda;
	
	LNKFloat *workspace = LNKFloatAlloc(_featureCount);
	
	for (LNKSize exampleIndex = 0; exampleIndex < exampleCount; exampleIndex++) {
		const LNKFloat *example = dataMatrix + _featureCount * exampleIndex;
		const LNKFloat *output = [outputMatrix exampleAtIndex:exampleIndex];
		const LNKFloat *indicator = [_indicatorMatrix exampleAtIndex:exampleIndex];
		
		LNKFloat *dataGradientLocation = dataGradient + exampleIndex * _featureCount;
		
		for (LNKSize userIndex = 0; userIndex < userCount; userIndex++) {
			if (indicator[userIndex]) {
				// inner = (X(example,:) . Theta(user,:)) - Y(example,user)
				const LNKFloat *user = thetaMatrix + userIndex * _featureCount;
				
				LNKFloat result;
				LNK_dotpr(example, UNIT_STRIDE, user, UNIT_STRIDE, &result, _featureCount);
				
				const LNKFloat inner = result - output[userIndex];
				
				// X_gradient += inner * Theta(user,:)
				LNKFloatCopy(workspace, user, _featureCount);
				LNK_vsmul(workspace, UNIT_STRIDE, &inner, workspace, UNIT_STRIDE, _featureCount);
				
				LNK_vadd(dataGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, dataGradientLocation, UNIT_STRIDE, _featureCount);
				
				// Theta_gradient += inner * X(example,:)
				LNKFloatCopy(workspace, example, _featureCount);
				LNK_vsmul(workspace, UNIT_STRIDE, &inner, workspace, UNIT_STRIDE, _featureCount);
				
				LNKFloat *thetaGradientLocation = thetaGradient + userIndex * _featureCount;
				LNK_vadd(thetaGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, thetaGradientLocation, UNIT_STRIDE, _featureCount);
			}
		}
		
		if (regularizationEnabled) {
			// X_gradient(example,:) += lambda * X(example,:)
			LNKFloatCopy(workspace, example, _featureCount);
			LNK_vsmul(workspace, UNIT_STRIDE, &lambda, workspace, UNIT_STRIDE, _featureCount);
			LNK_vadd(dataGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, dataGradientLocation, UNIT_STRIDE, _featureCount);
		}
	}
	
	if (regularizationEnabled) {
		for (LNKSize userIndex = 0; userIndex < userCount; userIndex++) {
			LNKFloat *thetaGradientLocation = thetaGradient + userIndex * _featureCount;
			
			// Theta_gradient(user,:) += lambda * Theta(user,:)
			LNKFloatCopy(workspace, thetaMatrix + userIndex * _featureCount, _featureCount);
			LNK_vsmul(workspace, UNIT_STRIDE, &lambda, workspace, UNIT_STRIDE, _featureCount);
			LNK_vadd(thetaGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, thetaGradientLocation, UNIT_STRIDE, _featureCount);
		}
	}
	
	free(workspace);
	
	return unrolledGradient;
}

- (void)loadThetaMatrix:(LNKMatrix *)thetaMatrix {
	NSParameterAssert(thetaMatrix);
	NSParameterAssert(thetaMatrix.columnCount == _featureCount);
	
	LNKMatrix *outputMatrix = self.matrix;
	const LNKSize userCount = outputMatrix.columnCount;
	const LNKSize exampleCount = outputMatrix.exampleCount;
	
	LNKFloatCopy(_unrolledGradient + _featureCount * exampleCount, thetaMatrix.matrixBuffer, userCount * thetaMatrix.columnCount);
}

- (void)loadDataMatrix:(LNKMatrix *)dataMatrix {
	NSParameterAssert(dataMatrix);
	NSParameterAssert(dataMatrix.columnCount == _featureCount);
	
	const LNKSize exampleCount = self.matrix.exampleCount;
	LNKFloatCopy(_unrolledGradient, dataMatrix.matrixBuffer, exampleCount * _featureCount);
}

static LNKCollaborativeFilteringPredictor *tempSelf = nil;

static void _fmincg_evaluate(LNKFloat *inputVector, LNKFloat *outCost, LNKFloat *gradientVector) {
	LNKCollaborativeFilteringPredictor *self = tempSelf;
	assert(self);
	assert(inputVector);
	assert(outCost);
	assert(gradientVector);
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize userCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize unrolledExampleCount = exampleCount + userCount;
	const LNKSize totalCount = unrolledExampleCount * self->_featureCount;
	
	LNKFloatCopy(self->_unrolledGradient, inputVector, totalCount);
	
	const LNKFloat cost = [self _evaluateCostFunction];
	const LNKFloat *gradient = [self _computeGradient];
	LNKFloatCopy(gradientVector, gradient, totalCount);
	free((void *)gradient);
	
	*outCost = cost;
}

- (void)train {
	tempSelf = self;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize userCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize unrolledExampleCount = exampleCount + userCount;
	const LNKSize totalCount = unrolledExampleCount * _featureCount;
	
	const LNKFloat epsilon = 1.5;
	
	for (LNKSize n = 0; n < totalCount; n++) {
		_unrolledGradient[n] = (((LNKFloat) arc4random_uniform(UINT32_MAX) / UINT32_MAX) - 0.5) * 2 * epsilon;
	}
	
#ifdef DEBUG
	int result = fmincg(_fmincg_evaluate, _unrolledGradient, (int)totalCount, (int)algorithm.iterationCount);
	NSAssert(result == 0 || result == 1, @"Could not minimize the function");
#else
	fmincg(_fmincg_evaluate, _unrolledGradient, (int)totalCount, (int)algorithm.iterationCount);
#endif
}

- (NSIndexSet *)findTopK:(LNKSize)k predictionsForUser:(LNKSize)userIndex {
	if (k == 0)
		[NSException raise:NSInvalidArgumentException format:@"The parameter k must be greater than 0"];
	
	LNKMatrix *outputMatrix = self.matrix;
	const LNKSize userCount = outputMatrix.columnCount;
	const LNKSize exampleCount = outputMatrix.exampleCount;
	
	const LNKFloat *dataMatrix = _unrolledGradient;
	const LNKFloat *thetaMatrix = _unrolledGradient + exampleCount * _featureCount;
	
	LNKFloat *predictions = LNKFloatAlloc(userCount * exampleCount);
	LNK_mmul(dataMatrix, UNIT_STRIDE, thetaMatrix, UNIT_STRIDE, predictions, UNIT_STRIDE, exampleCount, userCount, _featureCount);
	
	NSMutableArray *results = [NSMutableArray new];
	
	for (LNKSize example = 0; example < exampleCount; example++) {
		LNKFloat prediction = predictions[example * _featureCount + userIndex];
		
		[results addObject:@{ @"prediction": @(prediction), @"index": @(example) }];
	}
	
	free(predictions);
	[results sortUsingDescriptors:@[ [NSSortDescriptor sortDescriptorWithKey:@"prediction" ascending:NO] ]];
	
	NSMutableIndexSet *indices = [NSMutableIndexSet new];
	
	for (LNKSize n = 0; n < k; n++) {
		NSDictionary *result = results[n];
		[indices addIndex:[result[@"index"] unsignedIntegerValue]];
	}
	
	[results release];
	
	return [indices autorelease];
}

- (void)dealloc {
	free(_unrolledGradient);
	
	[_indicatorMatrix release];
	
	[super dealloc];
}

@end
