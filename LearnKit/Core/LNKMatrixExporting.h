//
//  LNKMatrixExporting.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

@interface LNKMatrix (Exporting)

- (BOOL)writeCSVDataToURL:(NSURL *)url error:(NSError **)outError;

@end
