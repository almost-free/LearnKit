//
//  LNKRandomForestClassifier.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <LearnKit/LNKClassifier.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, LNKRandomForestClassifierMaxFeatures) {
	LNKRandomForestClassifierMaxFeaturesSqrt,
	LNKRandomForestClassifierMaxFeaturesLog2
};

/// A random forest classifier for matrices of discrete values.
/// The optimization algorithm is ignored and can be `nil`.
/// The value types of the matrix must be registered prior to training.
/// Output labels must be specified in form of classes.
@interface LNKRandomForestClassifier : LNKClassifier

/// Determines how many features should be randomly selected when training a tree. The default is `sqrt(columnCount)`.
@property (nonatomic) LNKRandomForestClassifierMaxFeatures maxFeatures;

/// Determines how many trees should be built during training. The default is `10`.
@property (nonatomic) LNKSize treeCount;

/// Determines how many examples should be randomly selected when training a tree. A value of `0` indicates 'all'.
@property (nonatomic) LNKSize maxExampleCount;

/// Indicates values at `columnIndex` are boolean values.
- (void)registerBooleanValueForColumnAtIndex:(LNKSize)columnIndex;

/// Indicates values at `columnIndex` are categorical in range [0, valueCount).
- (void)registerCategoricalValues:(LNKSize)valueCount forColumnAtIndex:(LNKSize)columnIndex;

@end

NS_ASSUME_NONNULL_END
