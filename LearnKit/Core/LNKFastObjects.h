//
//  LNKFastObjects.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

typedef struct _LNKFastObject LNKFastObject;
typedef struct _LNKFastArray LNKFastArray;
typedef struct _LNKFastString LNKFastString;

typedef LNKFastObject *LNKFastObjectRef;
typedef LNKFastArray *LNKFastArrayRef;
typedef LNKFastString *LNKFastStringRef;

void LNKFastObjectRetain(LNKFastObjectRef object);
void LNKFastObjectRelease(LNKFastObjectRef object);

/// The returned object is reference-counted.
LNKFastArrayRef LNKFastArrayCreate();

/// Adds a fast object to the array (and increments its retain count).
void LNKFastArrayAddFastObject(LNKFastArrayRef array, LNKFastObjectRef object);

/// Returns the fast object at the given index.
LNKFastObjectRef LNKFastArrayObjectAtIndex(LNKFastArrayRef array, LNKSize index);

LNKSize LNKFastArrayObjectCount(LNKFastArrayRef array);

void LNKFastArrayRelease(LNKFastArrayRef);

/// The returned object is reference-counted.
LNKFastStringRef LNKFastStringCreateWithUTF8String(const char *, LNKSize);

const char *LNKFastStringGetUTF8String(LNKFastStringRef);
