//
//  LNKCollaborativeFilteringPredictorPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKCollaborativeFilteringPredictor.h"

@interface LNKCollaborativeFilteringPredictor (Private)

- (void)_copyThetaMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose;

@end
