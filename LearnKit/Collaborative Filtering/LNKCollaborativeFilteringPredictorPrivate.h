//
//  LNKCollaborativeFilteringPredictorPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKCollaborativeFilteringPredictor.h"

@interface LNKCollaborativeFilteringPredictor (Private)

/// The returned memory buffer must be freed.
- (const LNKFloat *)_computeGradient;

@end
