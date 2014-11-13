//
//  LNKClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

#import "LNKClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKMatrixPrivate.h"

@implementation LNKClass

+ (instancetype)classWithUnsignedInteger:(NSUInteger)value {
	return [[[self alloc] initWithUnsignedInteger:value] autorelease];
}

- (instancetype)initWithUnsignedInteger:(NSUInteger)value {
	if (!(self = [super init]))
		return nil;
	
	_unsignedIntegerValue = value;
	
	return self;
}

- (NSUInteger)hash {
	return _unsignedIntegerValue;
}

- (BOOL)isEqual:(LNKClass *)object {
	return [object isKindOfClass:[LNKClass class]] && _unsignedIntegerValue == object->_unsignedIntegerValue;
}

@end


@implementation LNKClasses {
	NSArray *_classes;
	LNKClassMapper _mapper;
}

static NSArray *_LNKIntegersInRange(NSRange range) {
	NSMutableArray *array = [[NSMutableArray alloc] init];
	
	for (NSUInteger n = range.location; n < NSMaxRange(range); n++) {
		[array addObject:[LNKClass classWithUnsignedInteger:n]];
	}
	
	return [array autorelease];
}

+ (instancetype)withRange:(NSRange)range {
	NSParameterAssert(range.length);
	NSArray *classes = _LNKIntegersInRange(range);
	
	return [self withClasses:classes mapper:^NSUInteger(LNKClass *aClass) {
		return [aClass unsignedIntegerValue] - range.location;
	}];
}

+ (instancetype)withCount:(LNKSize)count {
	NSParameterAssert(count);
	return [self withRange:NSMakeRange(0, count)];
}

+ (instancetype)withClasses:(NSArray *)classes mapper:(LNKClassMapper)mapper {
	return [[self alloc] initWithClasses:classes mapper:mapper];
}

- (instancetype)initWithClasses:(NSArray *)classes mapper:(LNKClassMapper)mapper {
	NSParameterAssert(classes.count);
	NSParameterAssert(mapper);
	
	if (!(self = [super init]))
		return nil;
	
	_classes = [classes retain];
	_mapper = [mapper copy];
	
	return self;
}

- (NSUInteger)indexForClass:(LNKClass *)aClass {
	NSParameterAssert(aClass);
	return _mapper(aClass);
}

- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(__unsafe_unretained id [])buffer count:(NSUInteger)len {
	return [_classes countByEnumeratingWithState:state objects:buffer count:len];
}

- (NSUInteger)count {
	return _classes.count;
}

- (void)dealloc {
	[_classes release];
	[_mapper release];
	[super dealloc];
}

@end


@implementation LNKClassifier {
	NSMapTable *_classesToProbabilities;
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	NSParameterAssert(classes);
	
	if (!(self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm]))
		return nil;
	
	_classes = [classes retain];
	
	return self;
}

- (void)_didPredictProbability:(LNKFloat)probability forClass:(LNKClass *)class {
	NSParameterAssert(class);
	
	if (!_classesToProbabilities)
		_classesToProbabilities = [[NSMapTable strongToStrongObjectsMapTable] retain];
	
	[_classesToProbabilities setObject:[NSNumber numberWithLNKFloat:probability] forKey:class];
}

- (void)_predictValueForFeatureVector:(LNKVector)featureVector {
#pragma unused(featureVector)
	
	NSAssertNotReachable(@"%s should be implemented by subclasses", __PRETTY_FUNCTION__);
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);
	
	if (featureVector.length != self.matrix.columnCount)
		[NSException raise:NSGenericException format:@"The length of the feature vector must be equal to the number of columns in the matrix"]; // otherwise, we can't do matrix multiplication
	
	[self _predictValueForFeatureVector:featureVector];
	
	LNKFloat bestProbability = -1;
	LNKClass *bestClass = nil;
	
	for (LNKClass *class in _classes) {
		LNKFloat probability = [[_classesToProbabilities objectForKey:class] LNKFloatValue];
		
		if (probability > bestProbability) {
			bestClass = class;
			bestProbability = probability;
		}
	}
	
	return bestClass;
}

- (LNKFloat)computeClassificationAccuracy {
	LNKMatrix *matrix = self.matrix;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKSize hits = 0;
	
	for (LNKSize m = 0; m < exampleCount; m++) {
		id predictedValue = [self predictValueForFeatureVector:LNKVectorMakeUnsafe(_EXAMPLE_IN_MATRIX_BUFFER(m), columnCount)];
		
		if ([predictedValue isEqual:[LNKClass classWithUnsignedInteger:outputVector[m]]])
			hits++;
	}
	
	return (LNKFloat)hits / exampleCount;
}

- (void)dealloc {
	[_classesToProbabilities release];
	[_classes release];
	[super dealloc];
}

@end
