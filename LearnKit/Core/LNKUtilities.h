//
//  LNKUtilities.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

void LNKPrintVector(const char *name, LNKFloat *vector, LNKSize n);
void LNKPrintMatrix(const char *name, LNKFloat *matrix, LNKSize m, LNKSize n);

/// The matrix data is loaded in column-major order.
NSData *__nullable LNKLoadBinaryMatrixFromFileAtURL(NSURL *url, LNKSize expectedLength);

NS_ASSUME_NONNULL_END
