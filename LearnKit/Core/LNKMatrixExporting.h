//
//  LNKMatrixExporting.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKMatrix (Exporting)

- (BOOL)writeCSVDataToURL:(NSURL *)url error:(NSError **)outError;

@end

NS_ASSUME_NONNULL_END
