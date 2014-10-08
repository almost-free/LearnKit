//
//  LNKFastArray.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

typedef struct _LNKFastArray LNKFastArray;
typedef LNKFastArray *LNKFastArrayRef;

/// The returned object must be freed with `LNKFastArrayFree`.
/// Each element stored in the array must be of size `elementWidth`.
extern LNKFastArrayRef LNKFastArrayCreate(LNKSize elementWidth);

/// The same fast array must not be freed more than once.
extern void LNKFastArrayFree(LNKFastArrayRef array);

/// Adds element at pointer `bytes` of element width to the array.
extern void LNKFastArrayAddElement(LNKFastArrayRef array, void *bytes);

/// Returns a pointer to the element at the given index. Its alloted memory is of element width size.
extern void *LNKFastArrayElementAtIndex(LNKFastArrayRef array, LNKSize index);

extern LNKSize LNKFastArrayElementCount(LNKFastArrayRef array);
