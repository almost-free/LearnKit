//
//  LNKCollaborativeFilteringPredictorPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKCollaborativeFilteringPredictor.h"

@interface LNKCollaborativeFilteringPredictor (Private)

- (void)_copyThetaVector:(const LNKFloat *)vector shouldTranspose:(BOOL)shouldTranspose;

@end
