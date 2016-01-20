//
//  LNKMatrixUI.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#if TARGET_OS_MAC

#import "LNKMatrixUI.h"
#import "Numbers.h"

#import <AppKit/AppKit.h>
#import <ScriptingBridge/ScriptingBridge.h>

@implementation LNKMatrix (UI)

- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height {
	if (!width || !height)
		[NSException raise:NSGenericException format:@"The width and height must be greater than 0"];
	
	if (index >= self.exampleCount)
		[NSException raise:NSGenericException format:@"The given index (%lld) is out-of-bounds (%lld)", index, self.exampleCount];
	
	const LNKFloat *pixels = self.matrixBuffer + (index * self.columnCount);
	
	return [NSImage imageWithSize:NSMakeSize(width, height) flipped:YES drawingHandler:^BOOL(NSRect dstRect) {
#pragma unused(dstRect)
		
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

- (void)importToNumbersAsTable {
	NumbersApplication *app = [SBApplication applicationWithBundleIdentifier:@"com.apple.iWork.Numbers"];
	
	NumbersDocument *document = [[[app classForScriptingClass:@"document"] alloc] init];
	[app.documents addObject:document];
	
	[document.sheets removeAllObjects];
	
	NumbersSheet *defaultSheet = [[[app classForScriptingClass:@"sheet"] alloc] init];
	[document.sheets addObject:defaultSheet];
	
	[defaultSheet.tables removeAllObjects];
	
	NumbersTable *table = [[[app classForScriptingClass:@"table"] alloc] init];
	[defaultSheet.tables addObject:table];
	
	table.headerColumnCount = 0;
	table.headerRowCount = 0;
	table.rowCount = self.exampleCount;
	table.columnCount = self.columnCount;
	table.name = @"Matrix";
	
	const LNKFloat *buffer = self.matrixBuffer;
	
	LNKSize offset = 0;
	
	for (NumbersCell *cell in table.cells) {
		cell.value = @(buffer[offset++]);
	}
	
	[table release];
	[defaultSheet release];
	[document release];
}

@end

#endif
