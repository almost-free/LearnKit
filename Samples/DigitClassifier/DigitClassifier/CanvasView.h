//
//  CanvasView.h
//  DigitClassifier
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <LearnKit/LearnKit.h>

@protocol CanvasViewDelegate

- (void)canvasViewRecognizeFeatureVector:(const LNKFloat *)vector;

@end

@interface CanvasView : NSView

- (void)loadFeatureVector:(const LNKFloat *)vector;

@property (nonatomic, weak) id <CanvasViewDelegate> delegate;

@end
