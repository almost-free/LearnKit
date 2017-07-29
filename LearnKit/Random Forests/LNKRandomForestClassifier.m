//
//  LNKRandomForestClassifier.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKRandomForestClassifier.h"

#import "LNKDecisionTreeClassifier.h"
#import "LNKDecisionTreeClassifier+Private.h"
#import "LNKMatrix.h"
#import "NSCountedSetAdditions.h"
#import "NSIndexSetAdditions.h"

@implementation LNKRandomForestClassifier {
	NSArray<LNKDecisionTreeClassifier *> *_trees;
	NSMutableDictionary<NSNumber *, NSNumber *> *_columnsToPossibleValues;
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[];
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)

	return [self class];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	if (!(self = [super initWithMatrix:matrix implementationType:implementationType optimizationAlgorithm:algorithm classes:classes])) {
		return nil;
	}

	_treeCount = 10;
	_columnsToPossibleValues = [[NSMutableDictionary alloc] init];

	return self;
}

- (void)dealloc {
	[_columnsToPossibleValues release];
	[super dealloc];
}

- (void)registerBooleanValueForColumnAtIndex:(LNKSize)columnIndex {
	[self registerCategoricalValues:2 forColumnAtIndex:columnIndex];
}

- (void)registerCategoricalValues:(LNKSize)valueCount forColumnAtIndex:(LNKSize)columnIndex {
	if (columnIndex >= self.matrix.columnCount) {
		[NSException raise:NSInvalidArgumentException format:@"`columnIndex` is out of bounds"];
	}

	_columnsToPossibleValues[@(columnIndex)] = @(valueCount);
}

- (void)train {
	if (_trees != nil) {
		[_trees release];
		_trees = nil;
	}

	LNKMatrix *const matrix = self.matrix;
	const LNKSize totalExampleCount = matrix.rowCount;
	const LNKSize totalFeatureCount = matrix.columnCount;
	const LNKSize exampleCount = _maxExampleCount == 0 ? totalExampleCount : _maxExampleCount;

	LNKSize featureCount;
	switch (_maxFeatures) {
	case LNKRandomForestClassifierMaxFeaturesLog2:
		featureCount = (LNKSize)ceil(log2((double)totalFeatureCount));
		break;
	case LNKRandomForestClassifierMaxFeaturesSqrt:
		featureCount = (LNKSize)ceil(sqrt((double)totalFeatureCount));
		break;
	}

	NSMutableArray<LNKDecisionTreeClassifier *> *const newTrees = [[NSMutableArray alloc] init];

	for (LNKSize index = 0; index < _treeCount; index++) {
		NSIndexSet *const exampleIndices = [[NSIndexSet withCount:totalExampleCount] indexSetByRandomlySamplingTo:exampleCount];
		NSIndexSet *const columnIndices = [[NSIndexSet withCount:totalFeatureCount] indexSetByRandomlySamplingTo:featureCount];

		LNKDecisionTreeClassifier *const classifier = [[LNKDecisionTreeClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:self.classes examplesIndices:exampleIndices columnIndices:columnIndices];
		[newTrees addObject:classifier];
		[classifier release];

		[classifier _setColumnsToPossibleValues:_columnsToPossibleValues];
		[classifier train];
	}

	_trees = [newTrees copy];
	[newTrees release];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	if (_trees == nil) {
		return nil;
	}

	NSCountedSet<LNKClass *> *classes = [[NSCountedSet alloc] initWithCapacity:(NSUInteger)_treeCount];
	for (LNKDecisionTreeClassifier *tree in _trees) {
		id value = [tree predictValueForFeatureVector:featureVector];
		if (![value isKindOfClass:[LNKClass class]]) {
			continue;
		}
		[classes addObject:(LNKClass *)value];
	}

	LNKClass *const mostFrequentClass = classes.mostFrequentObject;
	NSAssert(mostFrequentClass != nil, @"There should be a most frequent class always");

	[classes release];

	return mostFrequentClass;
}

@end
