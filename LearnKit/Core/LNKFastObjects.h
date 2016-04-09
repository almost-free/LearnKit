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

extern void LNKFastObjectRetain(LNKFastObjectRef object);
extern void LNKFastObjectRelease(LNKFastObjectRef object);

/// The returned object is reference-counted.
extern LNKFastArrayRef LNKFastArrayCreate();

/// Adds a fast object to the array (and increments its retain count).
extern void LNKFastArrayAddFastObject(LNKFastArrayRef array, LNKFastObjectRef object);

/// Returns the fast object at the given index.
extern LNKFastObjectRef LNKFastArrayObjectAtIndex(LNKFastArrayRef array, LNKSize index);

extern LNKSize LNKFastArrayObjectCount(LNKFastArrayRef array);

extern void LNKFastArrayRelease(LNKFastArrayRef);

/// The returned object is reference-counted.
extern LNKFastStringRef LNKFastStringCreateWithUTF8String(const char *, LNKSize);

extern const char *LNKFastStringGetUTF8String(LNKFastStringRef);
