//
//  LNKMatrixExporting.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrixExporting.h"

@implementation LNKMatrix (Exporting)

- (BOOL)writeCSVDataToURL:(NSURL *)url error:(NSError **)outError {
	if (!url)
		[NSException raise:NSInvalidArgumentException format:@"The url must not be nil"];
	
	NSMutableString *output = [[NSMutableString alloc] init];
	
	for (LNKSize exampleIndex = 0; exampleIndex < self.rowCount; exampleIndex++) {
		const LNKFloat *example = [self rowAtIndex:exampleIndex];
		
		for (LNKSize column = 0; column < self.columnCount; column++) {
			[output appendFormat:@"%g", example[column]];
			
			if ((NSInteger)column < (NSInteger)self.columnCount - 1) {
				[output appendString:@", "];
			}
		}
		
		[output appendString:@"\n"];
	}
	
	if (![output writeToURL:url atomically:YES encoding:NSUTF8StringEncoding error:outError]) {
		[output release];
		return NO;
	}
	
	[output release];
	
	return YES;
}

@end
