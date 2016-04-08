//
//  LNKMatrixCSV.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

NS_ASSUME_NONNULL_BEGIN

@class LNKCSVColumnRule;

@interface LNKMatrix (CSV)

/// Initializes a matrix by loading a CSV file. The file should not contain headings.
/// Optionally, a ones column may be added to the beginning of the matrix. The last column will be mapped to the output vector.
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url;
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter;

/// Columns may be deleted or transformed (e.g. mapping strings representing categorical data to numerical entries) by passing a dictionary of preprocessing rules, indexed by the column index.
- (nullable instancetype)initWithCSVFileAtURL:(NSURL *)url delimiter:(unichar)delimiter ignoringHeader:(BOOL)ignoreHeader columnPreprocessingRules:(NSDictionary<NSNumber *, LNKCSVColumnRule *> *)preprocessingRules;

@end

NS_ASSUME_NONNULL_END
