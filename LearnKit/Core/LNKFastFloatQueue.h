//
//  LNKFastFloatQueue.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

/// A fast queue with FIFO semantics that holds LNKFloat values.
typedef struct _LNKFastFloatQueue LNKFastFloatQueue;
typedef LNKFastFloatQueue *LNKFastFloatQueueRef;

/// The returned object must be freed with `LNKFastFloatQueueFree`.
/// Only `capacity` number of items can be stored in the queue.
LNKFastFloatQueueRef LNKFastFloatQueueCreate(LNKSize capacity);

/// The same queue must not be freed more than once.
void LNKFastFloatQueueFree(LNKFastFloatQueueRef queue);

void LNKFastFloatQueueEnqueue(LNKFastFloatQueueRef queue, LNKFloat value);
LNKFloat LNKFastFloatQueueDequeue(LNKFastFloatQueueRef queue);

LNKSize LNKFastFloatQueueSize(LNKFastFloatQueueRef queue);

BOOL LNKFastFloatAreValuesApproximatelyClose(LNKFastFloatQueueRef queue, LNKFloat threshold);
