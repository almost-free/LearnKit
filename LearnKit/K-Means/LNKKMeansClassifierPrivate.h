//
//  LNKKMeansClassifierPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKKMeansClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKKMeansClassifier (Private)

- (LNKFloat *)_clusterCentroids NS_RETURNS_INNER_POINTER;
- (void)_setClusterCentroids:(const LNKFloat *)clusterCentroids;

@end

NS_ASSUME_NONNULL_END
