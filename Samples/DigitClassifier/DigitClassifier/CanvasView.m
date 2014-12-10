//
//  CanvasView.m
//  DigitClassifier
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "CanvasView.h"

@implementation CanvasView {
	LNKFloat *_featureVector;
}

static const NSUInteger kMaxLength = 20;
static const CGFloat kBoxLength = 8;
static const CGFloat kCenteringFix = 0.5;

- (void)_drawGrid {
	const CGFloat boundsWidth = NSWidth(self.bounds);
	const CGFloat boundsHeight = NSHeight(self.bounds);
	
	[[NSColor gridColor] set];
	
	for (NSUInteger column = 0; column < kMaxLength; column++) {
		NSBezierPath *line = [NSBezierPath bezierPath];
		[line moveToPoint:NSMakePoint(column * kBoxLength + kCenteringFix, 0)];
		[line lineToPoint:NSMakePoint(column * kBoxLength + kCenteringFix, boundsHeight)];
		[line stroke];
	}
	
	for (NSUInteger row = 0; row < kMaxLength; row++) {
		NSBezierPath *line = [NSBezierPath bezierPath];
		[line moveToPoint:NSMakePoint(0, row * kBoxLength + kCenteringFix)];
		[line lineToPoint:NSMakePoint(boundsWidth, row * kBoxLength + kCenteringFix)];
		[line stroke];
	}
	
	if (!_featureVector)
		return;
	
	for (NSUInteger row = 0; row < kMaxLength; row++) {
		for (NSUInteger column = 0; column < kMaxLength; column++) {
			
			if (_featureVector[row * kMaxLength + column]) {
				NSRectFill(NSMakeRect(row * kBoxLength, column * kBoxLength, kBoxLength, kBoxLength));
			}
		}
	}
}

- (void)loadFeatureVector:(const LNKFloat *)vector {
	_featureVector = LNKFloatAllocAndCopy(vector, kMaxLength * kMaxLength);
	
	[self setNeedsDisplay:YES];
}

- (BOOL)isFlipped {
	return YES;
}

- (void)drawRect:(NSRect)dirtyRect {
	[self _drawGrid];
}

- (void)mouseDown:(NSEvent *)theEvent {
	if (_featureVector)
		free(_featureVector);
	
	_featureVector = LNKFloatCalloc(kMaxLength * kMaxLength);
	
	[self setNeedsDisplay:YES];
}

- (void)mouseDragged:(NSEvent *)theEvent {
	const NSPoint localPoint = [self convertPoint:theEvent.locationInWindow fromView:nil];
	
	const NSUInteger row = localPoint.y / kBoxLength;
	const NSUInteger column = localPoint.x / kBoxLength;
	
	const NSUInteger index = column * kMaxLength + row;
	_featureVector[index] = 1;
	
	[self setNeedsDisplay:YES];
}

- (void)mouseUp:(NSEvent *)theEvent {
	[self.delegate canvasViewRecognizeFeatureVector:_featureVector];
}

@end
