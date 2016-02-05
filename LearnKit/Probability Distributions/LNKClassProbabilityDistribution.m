//
//  LNKClassProbabilityDistribution.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKClassProbabilityDistribution.h"

#import "LNKClasses.h"
#import "LNKClassProbabilityDistributionPrivate.h"

@implementation LNKClassProbabilityDistribution {
	LNKFloat *_priorProbabilities;
}

- (instancetype)initWithClasses:(LNKClasses *)classes featureCount:(LNKSize)featureCount {
	if (featureCount < 1) {
		[NSException raise:NSInvalidArgumentException format:@"At least one feature must be specified"];
	}

	if (classes.count < 2) {
		[NSException raise:NSInvalidArgumentException format:@"At least two classes must be given"];
	}

	if (!(self = [super init]))
		return nil;

	_classes = [classes retain];
	_featureCount = featureCount;
	_priorProbabilities = LNKFloatAlloc(classes.count);

	return self;
}

- (void)dealloc {
	if (_priorProbabilities != NULL) {
		free(_priorProbabilities);
	}

	[_classes release];

	[super dealloc];
}

- (void)buildWithMatrix:(LNKMatrix *)matrix {
#pragma unused(matrix)
	[NSException raise:NSGenericException format:@"%s must be overriden", __PRETTY_FUNCTION__];
}

- (LNKFloat)probabilityForClassAtIndex:(LNKSize)classIndex featureAtIndex:(LNKSize)featureIndex value:(LNKFloat)value {
#pragma unused(classIndex)
#pragma unused(featureIndex)
#pragma unused(value)
	[NSException raise:NSGenericException format:@"%s must be overriden", __PRETTY_FUNCTION__];
	return 0;
}

- (LNKFloat)priorForClassAtIndex:(LNKSize)index {
	if (index >= self.classes.count) {
		[NSException raise:NSInvalidArgumentException format:@"The class index is out-of-bounds"];
	}

	return _priorProbabilities[index];
}

- (void)_setPrior:(LNKFloat)prior forClassAtIndex:(LNKSize)index {
	if (index >= self.classes.count) {
		[NSException raise:NSInvalidArgumentException format:@"The class index is out-of-bounds"];
	}

	_priorProbabilities[index] = prior;
}

@end
