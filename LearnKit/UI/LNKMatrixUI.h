//
//  LNKMatrixUI.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#if TARGET_OS_MAC

#import "LNKMatrix.h"

@interface LNKMatrix (UI)

/// Generates a grayscale image representing an example in the matrix.
/// Matrix values are treated as pixel intensities and should be normalized in range [0, 1].
- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height;

/// Creates a new Numbers document with a table representation of the matrix.
- (void)importToNumbersAsTable;

@end

#endif
