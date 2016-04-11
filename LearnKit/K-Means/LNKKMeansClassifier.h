//
//  LNKKMeansClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

/// The optimization algorithm for k-means classifiers is ignored and can be `nil`.
/// The classes specified correspond to the number of clusters.
/// For example, initializing a LNKClassifier with `[LNKClasses withCount:3]` specifies 3 clusters.
/// Predicted values are of type NSNumber/LNKSize and indicate the index of the closest cluster.
@interface LNKKMeansClassifier : LNKClassifier

/// The iteration count must be >= 1; the default is 100.
/// A value of `LNKSizeMax` may be specified to run the algorithm until convergence.
@property (nonatomic) LNKSize iterationCount;

/// The greatest tolerable distance between a data point and its cluster. Points at greater distances
/// will be assigned to a 'junk' cluster during iteration.
/// The default is `LNKFloatMax`, which effectively turns off the junk cluster.
@property (nonatomic) LNKFloat maximumClusterDistance;

/// The returned vector is +1 reference counted.
- (LNKVector)centroidForClusterAtIndex:(LNKSize)clusterIndex;

@end

NS_ASSUME_NONNULL_END
