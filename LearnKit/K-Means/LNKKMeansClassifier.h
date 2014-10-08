//
//  LNKKMeansClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// The optimization algorithm for k-means classifiers is ignored and can be `nil`.
/// The classes specified correspond to the number of clusters.
/// For example, initializing a LNKClassifier with `[LNKClasses withCount:3]` specifies 3 clusters.
/// Predicted values are of type NSNumber/LNKSize and indicate the index of the closest cluster.
@interface LNKKMeansClassifier : LNKClassifier

/// The iteration count must be >= 1; the default is 100.
@property (nonatomic) LNKSize iterationCount;

- (const LNKFloat *)centroidForClusterAtIndex:(LNKSize)clusterIndex;

@end
