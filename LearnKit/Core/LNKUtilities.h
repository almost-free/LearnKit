//
//  LNKUtilities.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

extern void LNKPrintVector(const char *name, LNKFloat *vector, LNKSize n);
extern void LNKPrintMatrix(const char *name, LNKFloat *matrix, LNKSize m, LNKSize n);

/// The matrix data is loaded in column-major order.
extern NSData *LNKLoadBinaryMatrixFromFileAtURL(NSURL *url, LNKSize expectedLength);
