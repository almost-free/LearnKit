//
//  LNKDesignMatrixUI.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDesignMatrixUI.h"

#import <AppKit/AppKit.h>

@implementation LNKDesignMatrix (UI)

#if TARGET_OS_MAC
- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height {
	NSParameterAssert(width);
	NSParameterAssert(height);
	NSParameterAssert(index < self.exampleCount);
	
	const LNKFloat *pixels = self.matrixBuffer + (index * self.columnCount);
	
	return [NSImage imageWithSize:NSMakeSize(width, height) flipped:YES drawingHandler:^BOOL(NSRect dstRect) {
		NSAssert(NSEqualRects(dstRect, NSMakeRect(0, 0, width, height)), @"Unexpected drawing area");
		
		for (LNKSize x = 0; x < width; x++) {
			for (LNKSize y = 0; y < height; y++) {
				const LNKFloat value = pixels[x * height + y];
				[[NSColor colorWithCalibratedWhite:value alpha:1] set];
				NSRectFill(NSMakeRect(x, y, 1, 1));
			}
		}
		
		return YES;
	}];
}
#endif

@end
