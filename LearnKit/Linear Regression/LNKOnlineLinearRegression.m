//
//  LNKOnlineLinearRegression.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKOnlineLinearRegression.h"

@implementation LNKOnlineLinearRegression {
	LNKSize _exampleCount;
	LNKFloat _sumX, _sumY;
	LNKFloat _sumXX, _sumYY, _sumXY;
}

- (void)addExampleWithX:(LNKFloat)x y:(LNKFloat)y
{
	_sumX += x;
	_sumY += y;
	_sumXX += x * x;
	_sumYY += y * y;
	_sumXY += x * y;
	_exampleCount++;
}

- (void)regress
{
	if (_exampleCount < 2) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"At least two training examples must be provided!" userInfo:nil];
	}

	LNKFloat sxx = _sumXX - _sumX * _sumX / _exampleCount;
	LNKFloat sxy = _sumXY - _sumX * _sumY / _exampleCount;
	_slope = sxy / sxx;
	_intercept = (_sumY - _slope * _sumX) / _exampleCount;
}

- (LNKFloat)predictYForX:(LNKFloat)x
{
	if (_exampleCount < 2) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"At least two training examples must be provided!" userInfo:nil];
	}
	
	return _intercept + _slope * x;
}

@end
