//
//  LNKAccelerateGradient.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAccelerateGradient.h"

#import "lbfgs.h"
#import "LNKAccelerate.h"
#import "LNKFastFloatQueue.h"
#import "LNKMatrixPrivate.h"

@interface LBFGSContext : NSObject

@property (nonatomic, assign) LNKMatrix *matrix;
@property (nonatomic) LNKFloat *thetaVector;
@property (nonatomic, copy) LNKHFunction hFunction;
@property (nonatomic, copy) LNKCostFunction costFunction;
@property (nonatomic) LNKFloat *workgroupCC;
@property (nonatomic) LNKFloat *workgroupCC2;
@property (nonatomic) LNKFloat *workgroupEC;
@property (nonatomic) const LNKFloat *transposeMatrix;
@property (nonatomic) BOOL regularizationEnabled;
@property (nonatomic) LNKFloat lambda;

@end

@implementation LBFGSContext
@end


// The final result is in workgroupCC.
void _LNKComputeBatchGradient(const LNKFloat *matrixBuffer, const LNKFloat *transposeMatrix, const LNKFloat *thetaVector, const LNKFloat *outputVector, LNKFloat *workgroupEC, LNKFloat *workgroupCC, LNKFloat *workgroupCC2, LNKSize exampleCount, LNKSize columnCount, BOOL enableRegularization, LNKFloat lambda, LNKHFunction hFunction) {
	// h = x . thetaVector
	// 1 / m * sum((h - y) * x)
	LNK_mmul(matrixBuffer, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroupEC, UNIT_STRIDE, exampleCount, 1, columnCount);
	
	if (hFunction)
		hFunction(workgroupEC, exampleCount);
	
	LNK_vsub(outputVector, UNIT_STRIDE, workgroupEC, UNIT_STRIDE, workgroupEC, UNIT_STRIDE, exampleCount);
	LNK_mmul(transposeMatrix, UNIT_STRIDE, workgroupEC, UNIT_STRIDE, workgroupCC, UNIT_STRIDE, columnCount, 1, exampleCount);
	
	if (enableRegularization) {
		// cost += lambda * theta
		LNKFloatCopy(workgroupCC2, thetaVector, columnCount);
		LNK_vsmul(workgroupCC2, UNIT_STRIDE, &lambda, workgroupCC2, UNIT_STRIDE, columnCount);
		
		// Don't regularize the first parameter.
		workgroupCC2[0] = 0;
		LNK_vadd(workgroupCC, UNIT_STRIDE, workgroupCC2, UNIT_STRIDE, workgroupCC, UNIT_STRIDE, columnCount);
	}
	
	const LNKFloat factor = 1.0 / exampleCount;
	LNK_vsmul(workgroupCC, UNIT_STRIDE, &factor, workgroupCC, UNIT_STRIDE, columnCount);
}

void LNK_learntheta_gd(LNKMatrix *matrix, LNKFloat *thetaVector, LNKOptimizationAlgorithmGradientDescent *algorithm, LNKCostFunction costFunction) {
	NSCAssert(matrix, @"The matrix must not be NULL");
	NSCAssert(thetaVector, @"The theta vector must not be NULL");
	NSCAssert(algorithm, @"The algorithm must not be nil");
	
	const LNKSize iterationCount = algorithm.iterationCount;
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = algorithm.lambda;
	
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKFloat *workgroupEC = LNKFloatAlloc(exampleCount);
	LNKFloat *workgroupCC = LNKFloatAlloc(columnCount);
	LNKFloat *workgroupCC2 = LNKFloatAlloc(columnCount);
	
	LNKFloat *transposeMatrix = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mtrans(matrixBuffer, transposeMatrix, columnCount, exampleCount);
	
	const BOOL stochastic = [algorithm isKindOfClass:[LNKOptimizationAlgorithmStochasticGradientDescent class]];
	
	void (^gradientIteration)(LNKFloat alpha) = ^(LNKFloat alpha) {
		if (stochastic) {
			LNKMatrix *randomMatrix = [matrix copyShuffledMatrix];
			const LNKFloat *randomMatrixBuffer = randomMatrix.matrixBuffer;
			const LNKSize stepCount = ((LNKOptimizationAlgorithmStochasticGradientDescent *)algorithm).stepCount;
			
			// Stochastic gradient descent:
			for (LNKSize step = 0; step < stepCount; step++) {
				const LNKFloat *row = randomMatrixBuffer + step * columnCount;
				
				// singleGradient = (h - y) * x
				LNKFloat h;
				LNK_dotpr(row, UNIT_STRIDE, thetaVector, UNIT_STRIDE, &h, columnCount);
				
				// workgroupCC holds the gradient.
				const LNKFloat delta = h - outputVector[step];
				LNK_vsmul(row, UNIT_STRIDE, &delta, workgroupCC, UNIT_STRIDE, columnCount);
				
				// thetaVector = thetaVector - alpha * gradient
				LNK_vsmul(workgroupCC, UNIT_STRIDE, &alpha, workgroupCC, UNIT_STRIDE, columnCount);
				LNK_vsub(workgroupCC, UNIT_STRIDE, thetaVector, UNIT_STRIDE, thetaVector, UNIT_STRIDE, columnCount);
			}

			[randomMatrix release];
		}
		else {
			// Batch gradient descent:
			// workgroupCC holds the gradient.
			_LNKComputeBatchGradient(matrixBuffer, transposeMatrix, thetaVector, outputVector, workgroupEC, workgroupCC, workgroupCC2, exampleCount, columnCount, regularizationEnabled, lambda, NULL);
			
			// thetaVector = thetaVector - alpha * gradient
			LNK_vsmul(workgroupCC, UNIT_STRIDE, &alpha, workgroupCC, UNIT_STRIDE, columnCount);
			LNK_vsub(workgroupCC, UNIT_STRIDE, thetaVector, UNIT_STRIDE, thetaVector, UNIT_STRIDE, columnCount);
		}
	};
	
	id <LNKAlpha> alphaBox = algorithm.alpha;
	const BOOL alphaIsDecaying = [alphaBox isKindOfClass:[LNKDecayingAlpha class]];
	LNKFloat alpha = alphaIsDecaying ? 0 : [(LNKFixedAlpha *)alphaBox value];
	
	if (iterationCount != NSNotFound) {
		for (LNKSize iteration = 0; iteration < iterationCount; iteration++) {
			if (alphaIsDecaying)
				alpha = [(LNKDecayingAlpha *)alphaBox function](iteration);
			
			gradientIteration(alpha);
		}
	}
	else {
		if (!costFunction)
			@throw [NSException exceptionWithName:NSGenericException reason:@"The cost function must be specified when automatically checking for convergence" userInfo:nil];
		
		static const LNKSize queueSize = 10;
		const LNKFloat convergenceThreshold = algorithm.convergenceThreshold;
		LNKFastFloatQueueRef costQueue = LNKFastFloatQueueCreate(queueSize);
		LNKSize iteration = 0;
		
		while (YES) {
			if (alphaIsDecaying)
				alpha = [(LNKDecayingAlpha *)alphaBox function](iteration++);
			
			gradientIteration(alpha);
			const LNKFloat cost = costFunction(thetaVector);
			
			if (LNKFastFloatQueueSize(costQueue) < queueSize)
				LNKFastFloatQueueEnqueue(costQueue, cost);
			else {
				if (LNKFastFloatAreValuesApproximatelyClose(costQueue, convergenceThreshold))
					break;
				
				LNKFastFloatQueueDequeue(costQueue);
				LNKFastFloatQueueEnqueue(costQueue, cost);
			}
		}
		
		LNKFastFloatQueueFree(costQueue);
	}
	
	free(workgroupEC);
	free(workgroupCC);
	free(workgroupCC2);
	free(transposeMatrix);
}

static lbfgsfloatval_t _LNK_lbfgs_evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {
#pragma unused(step)
#pragma unused(n)
	
	LBFGSContext *context = (__bridge LBFGSContext *)instance;
	NSCAssert(context, @"The context must not be nil");
	
	LNKMatrix *matrix = context.matrix;
	LNKFloat *workgroupCC = context.workgroupCC;
	LNKFloat *workgroupCC2 = context.workgroupCC2;
	const LNKSize columnCount = matrix.columnCount;
	NSCAssert(columnCount == (LNKSize)n, @"Size mismatch");
	
	_LNKComputeBatchGradient(matrix.matrixBuffer, context.transposeMatrix, x, matrix.outputVector, context.workgroupEC, workgroupCC, workgroupCC2, matrix.exampleCount, columnCount, context.regularizationEnabled, context.lambda, context.hFunction);
	
	// Give liblbfgs our gradient and return the cost.
	LNKFloatCopy(g, workgroupCC, columnCount);
	return context.costFunction(x);
}

void LNK_learntheta_lbfgs(LNKMatrix *matrix, LNKFloat *thetaVector, BOOL regularizationEnabled, LNKFloat lambda, LNKHFunction hFunction, LNKCostFunction costFunction) {
	NSCAssert(matrix, @"The matrix must not be NULL");
	NSCAssert(thetaVector, @"The theta vector must not be NULL");
	NSCAssert(costFunction, @"The cost function must not be nil");

	//TODO: should this be a static assert?
	NSCAssert(sizeof(lbfgsfloatval_t) == sizeof(LNKFloat), @"Size mismatch");
	
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	
	// Minimizing theta
	lbfgsfloatval_t *theta = calloc(columnCount, sizeof(lbfgsfloatval_t));
	
	lbfgs_parameter_t parameters;
	lbfgs_parameter_init(&parameters);
	
	// Lowered from 1e-5 for performance
	parameters.epsilon = 1e-4;
	
	LNKFloat *workgroupEC = LNKFloatAlloc(exampleCount);
	LNKFloat *workgroupCC = LNKFloatAlloc(columnCount);
	LNKFloat *workgroupCC2 = LNKFloatAlloc(columnCount);
	
	LNKFloat *transposeMatrix = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mtrans(matrix.matrixBuffer, transposeMatrix, columnCount, exampleCount);
	
	LBFGSContext *context = [[LBFGSContext alloc] init];
	context.matrix = matrix;
	context.thetaVector = thetaVector;
	context.hFunction = hFunction;
	context.costFunction = costFunction;
	context.workgroupEC = workgroupEC;
	context.workgroupCC = workgroupCC;
	context.workgroupCC2 = workgroupCC2;
	context.transposeMatrix = transposeMatrix;
	context.regularizationEnabled = regularizationEnabled;
	context.lambda = lambda;

#if DEBUG
	int status = lbfgs((int)columnCount, theta, NULL, _LNK_lbfgs_evaluate, NULL, (__bridge void *)context, &parameters);
	NSCAssert(status == 0, @"LBFGS optimization failed");
#else
	lbfgs((int)columnCount, theta, NULL, _LNK_lbfgs_evaluate, NULL, (__bridge void *)context, &parameters);
#endif

	[context release];
	
	free(workgroupEC);
	free(workgroupCC);
	free(workgroupCC2);
	free(transposeMatrix);
	
	free(theta);
}
