//
//  LNKMatrix+LinearRegressionAdditions.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrix+LinearRegressionAdditions.h"

#import "LNKAccelerate.h"

@implementation LNKMatrix (LinearRegressionAdditions)

- (void)performBoxCoxTransformationWithLambda:(LNKFloat)lambda
{
	[self modifyOutputVector:^(LNKFloat *outputVector, LNKSize m) {
		for (LNKSize i = 0; i < m; i++) {
			if (LNK_fabs(lambda) < LNKFloatMin) {
				outputVector[i] = LNKLog(outputVector[i]);
			}
			else {
				outputVector[i] = (LNK_pow(outputVector[i], lambda) - 1) / lambda;
			}
		}
	}];
}

@end
